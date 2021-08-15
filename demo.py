from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import numpy as np
import torch
import torch.nn.functional as F


tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
#model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

input = """PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""

#input_ids = tokenizer(input, return_tensors="pt").input_ids  # Batch size 1
#decoder_input_ids = tokenizer("California\'s", return_tensors="pt").input_ids  # Batch size 1
#decoder_input_ids = torch.LongTensor([[0, 222]])
#print(decoder_input_ids)


#shape = list(decoder_input_ids.size())
#shape[-1] = 1
#starts = torch.zeros(shape, dtype=torch.int)
#decoder_input_ids = torch.cat((starts, decoder_input_ids), -1)
#print(decoder_input_ids)

#shape = list(decoder_input_ids.size())
#last_dim = shape[-1]
#decoder_input_ids = torch.split(decoder_input_ids, [last_dim-1, 1], -1)[0]
#print(decoder_input_ids)

#logits = model.forward(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits
#print(logits)

#with torch.no_grad():
#    prob = F.softmax(logits, dim=2)
#print(prob)

#g = np.argmax(prob.detach().numpy(), axis=2)
#print(g)
#g = 222
print(tokenizer.decode([0, 5634, 4754, 6755, 50764, 35505], skip_special_tokens=True, clean_up_tokenization_spaces=False))
