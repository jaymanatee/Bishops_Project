import torch
from torchcrf import CRF
from transformers import BertModel, BertTokenizer
import os
from dotenv import load_dotenv
from groq import Groq


# Load the tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def destokenize(tensor):
    """
    Converts a tensor of token IDs into a readable string by reversing tokenization.
    Removes padding tokens.

    Args:
        tensor (torch.Tensor): A tensor containing token IDs.

    Returns:
        str: A decoded natural language string.
    """
    tokens = tokenizer.convert_ids_to_tokens(tensor.tolist())
    return tokenizer.convert_tokens_to_string(
        [token for token in tokens if token != "[PAD]"]
    )


def get_alert(text):
    """
    Extracts the alert string starting with 'ALERT->' and ending before the last 4 characters.
    Used to clean up the response from the language model.

    Args:
        text (str): The full text response from the model.

    Returns:
        str: Extracted alert text or None if not found.
    """
    index = text.rfind("ALERT->")
    if index != -1:
        return text[index:-4].strip()


class MyModel(torch.nn.Module):
    """
    A multitask model combining NER (Named Entity Recognition) with CRF and Sentiment Analysis (SA).
    Uses BERT embeddings, a shared BiLSTM layer, and two task-specific heads (NER and SA).
    """

    def __init__(self, hidden_dim, ner_output_dim, dropout_prob=0.25):
        super().__init__()

        self.embedder = BertModel.from_pretrained(
            "bert-base-uncased", ignore_mismatched_sizes=True
        )

        # Shared BiLSTM layer
        self.shared_lstm = torch.nn.LSTM(
            input_size=768, hidden_size=hidden_dim, batch_first=True, bidirectional=True
        )

        self.dropout = torch.nn.Dropout(dropout_prob)
        self.ln_lstm = torch.nn.LayerNorm(hidden_dim * 2)

        # Sentiment Analysis (SA) head
        self.sa_fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_dim, 1),
        )

        # Named Entity Recognition (NER) head
        self.ner_fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(hidden_dim, ner_output_dim),
        )

        # CRF for decoding NER tags
        self.crf = CRF(num_tags=ner_output_dim, batch_first=True)

    def forward(self, inputs):
        """
        Forward pass for the multitask model.

        Args:
            inputs (torch.Tensor): Input token IDs.

        Returns:
            ner_output (torch.Tensor): Raw emissions for NER.
            ner_output_decoded (torch.Tensor): CRF-decoded NER tags.
            sa_output (torch.Tensor): Predicted sentiment score (before thresholding).
        """
        if inputs is None:
            raise ValueError("Input tensor cannot be None")

        if isinstance(inputs, torch.Tensor):
            embedder_output = self.embedder(inputs)

        # Get contextual embeddings from BERT
        with torch.no_grad():
            embedder_output = self.embedder(inputs)

        embeddings = embedder_output.last_hidden_state

        # BiLSTM processing
        lstm_out, (h_n, _) = self.shared_lstm(embeddings)
        lstm_out = self.ln_lstm(self.dropout(lstm_out))

        # Get final hidden state for SA
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        final_hidden = torch.cat((h_forward, h_backward), dim=1)
        final_hidden = self.dropout(final_hidden)
        sa_output = self.sa_fc(final_hidden)

        # Get emission scores for CRF from NER head
        emissions = self.ner_fc(lstm_out)

        ner_output = emissions
        ner_output_decoded = self.crf.decode(emissions)
        ner_output_decoded = torch.stack(
            [torch.tensor(seq) for seq in ner_output_decoded]
        )

        return ner_output, ner_output_decoded, sa_output.squeeze(1)


class MyDeepSeek:
    """
    Wrapper class for generating alert messages using a language model (DeepSeek).
    Takes processed inputs and queries the model for natural language alerts.
    """

    def __init__(self, inputs, show_last=False):
        # Initialize Groq (DeepSeek) client
        load_dotenv()
        api_key = os.getenv("key")
        self.client = Groq(api_key=api_key)

        self.data = inputs
        self.show_last = show_last
        self.alerts = []

    def generate_alerts(self):
        """
        Generates alerts based on tweet, caption, NER, and sentiment.
        Only the final alert message is extracted and stored.

        Returns:
            List[str]: List of generated alerts.
        """
        for tweet_tensor, caption_tensor, ner_tags, sa_label in self.data:
            tweet_text = destokenize(tweet_tensor)
            caption_text = destokenize(caption_tensor)

            # Construct prompt for the LLM
            prompt = f"""
                You are a helpful assistant. Your task is to generate an alert based on the following information. Only a short and concise alert will be needed as the result of the prompt, nothing more.

                Tweet: {tweet_text}
                Caption: {caption_text}
                Named Entities (NER): {ner_tags}
                Sentiment: {"Positive" if sa_label == 1 else "Negative"}
                
                Based on the provided tweet, caption, named entities (NER), and sentiment, generate an alert following these indications:
                - The alert must start with "ALERT->".
                - The alert should be short, direct, and include the sentiment analysis (either "Positive" or "Negative").
                - If relevant named entities (NER) are present, include them in the alert.
                - If the sentiment is positive, the alert should reflect positive information; if the sentiment is negative, the alert should reflect negative information.
                - Always include the sentiment analysis at the end of the alert.

                ONLY generate the alert, NO other text. The result should only be the alert in the format:
                ALERT-> [Alert message] [Sentiment].
                """

            # Get streaming response from DeepSeek
            completion = self.client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_completion_tokens=4096,
                top_p=0.95,
                stream=True,
                stop=None,
            )

            result = ""
            for chunk in completion:
                result += f"{chunk.choices[0].delta.content}"

            # Extract alert from the model's response
            last_alert = get_alert(result)
            if last_alert:
                self.alerts.append(last_alert)

        if self.show_last:
            print("Tweet:", tweet_text)
            print("Caption:", caption_text)
            print("NER:", ner_tags)
            print("SA:", "Postive" if sa_label == 1 else "Negative")
            print(last_alert)

        return self.alerts
