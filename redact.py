# pip install presidio-analyzer presidio-anonymizer spacy langchain langchain-openai
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.predefined_recognizers import CreditCardRecognizer, EmailRecognizer, PhoneRecognizer, SpacyRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
import pandas as pd



class Redactor:

    def __init__(self):

        # Initialize the NLP engine
        self.provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_md"}]
        })
        self.nlp_engine = self.provider.create_engine()

        # Initialize the analyzer engine with the NLP engine
        self.analyzer = AnalyzerEngine(nlp_engine=self.nlp_engine, supported_languages=["en"])

        # Initialize the anonymizer engine
        self.anonymizer = AnonymizerEngine()


    def redact_pii(self, text):

        # Analyze the text to find PII entities
        results = self.analyzer.analyze(text=text, language="en")

        # Anonymize the detected PII entities
        anonymized_text = self.anonymizer.anonymize(text=text, analyzer_results=results)

        # return text, anonymized_text.text
        return anonymized_text.text


# def main():
#     # Read the CSV file
#     print("Reading CSV file...")
#     df = pd.read_csv('transcripts.csv')
    
#     # Initialize the redactor
#     redactor = Redactor()
    
#     # Create a new column for the redacted transcripts
#     df['redacted_transcript'] = None
#       # Process each row in the dataframe
#     print("Redacting PII from transcripts...")
#     for index, row in df.iterrows():
#         transcript = row['transcript']
#         original, redacted = redactor.redact_pii(transcript)
#         df.at[index, 'redacted_transcript'] = redacted
        
#         # Print progress for each row
#         print(f"Processed {index + 1}/{len(df)} transcripts")
    
#     # Save the redacted transcripts to a new CSV file
#     print("Saving redacted transcripts to CSV...")
#     df.to_csv('redacted_transcripts.csv', index=False)
#     print("Redaction complete. Redacted file saved as 'redacted_transcripts.csv'")


# if __name__ == "__main__":
#     main()

