import os
import argparse
import dataprofiler as dp

# Take filename of text file as an argument
parser = argparse.ArgumentParser()
parser.add_argument('text_file', help='text file to be redacted')
args = parser.parse_args()

# Process text file through DataLabeler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable TensorFlow info msgs
data_labeler = dp.DataLabeler(labeler_type='unstructured')
file_text = dp.Data(args.text_file)       # data/txt/text_file.txt
# Find the PII using the built-in model
model_predictions = data_labeler.predict(
    file_text, predict_options=dict(show_confidences=True))
# final_results = model_predictions["pred"]
# Set the output to the NER format (start position, end position, label)
data_labeler.set_params(
    {'postprocessor': {'output_format': 'ner',
                       'use_word_level_argmax': True}})
results = data_labeler.predict(file_text)

# Display the text file with redactions applied
orig_text = file_text.data[0]
redact_text = ''
pos = 0
pii = results['pred'][0]
for r in pii:
    redact_text += orig_text[pos:r[0]] + r[2]
    pos = r[1]
redact_text += orig_text[pos:]
print(redact_text)
