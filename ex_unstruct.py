import os
import dataprofiler as dp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable TensorFlow info msgs

data_labeler = dp.DataLabeler(labeler_type='unstructured')
my_text = dp.Data('data/txt/text_file.txt')

# Prediction what class each character belongs to
model_predictions = data_labeler.predict(
    my_text, predict_options=dict(show_confidences=True))

# Predictions / confidences are at the character level
final_results = model_predictions["pred"]
final_confidences = model_predictions["conf"]

# Set the output to the NER format (start position, end position, label)
data_labeler.set_params(
    {'postprocessor': {'output_format': 'ner',
                       'use_word_level_argmax': True}}
                       )
results = data_labeler.predict(my_text)

print(my_text.data)
print(results)
orig_text = my_text.data[0]
print(type(orig_text))
redact_text = ''
pos = 0
pii = results['pred'][0]
for r in pii:
    redact_text += orig_text[pos:r[0]] + r[2]
    pos = r[1]
redact_text += orig_text[pos:]
print(redact_text)
