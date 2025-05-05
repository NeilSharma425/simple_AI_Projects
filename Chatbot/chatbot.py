from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "facebook/blenderbot-400M-distill"
conversation_history = []
#convert history to string that is readable by tokenizer
history_string = "\n".join(conversation_history)
 #load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#load model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Tokenize model inputs

def callModel(minput):
    input_text = minput
    full_input = history_string + "\n" + input_text if history_string else input_text
    inputs = tokenizer(full_input, return_tensors="pt")
    print(inputs)

    # Generate Output
    outputs = model.generate(**inputs)
    print(outputs)

    #Decode output
    response = tokenizer.decode(outputs[0],skip_special_tokens=True).strip()
    print(response)

    conversation_history.append(input_text)
    conversation_history.append(response)
    print(conversation_history)
    return response