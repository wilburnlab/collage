import torch
from collage.model import Codon_Predictor
import torch.profiler

model = Codon_Predictor( n_input_tokens = 22, 
                         n_output_tokens = 66,
                         n_sp = 1, #len( species ),
                         model_dim = 64, 
                         ff_dim = 256,
                         n_heads = 4,
                         n_encoder_layers = 1, 
                         n_decoder_layers = 2, 
                         dropout = 0.2,
                         max_len = 500, )

state_dict = torch.load('test.pt')
model.load_state_dict(state_dict)

prot = torch.tensor([[14, 13,  6,  6, 11,  1,  3, 14, 18,  1,  4, 16, 10, 11,  1,  1,  1, 15,
         15,  6,  8,  7,  2, 15, 10, 11, 10,  3, 16,  1],
        [ 3, 10,  7,  1,  1,  6,  6,  1,  3, 12, 13,  1,  6, 18,  3,  3,  9,  3,
         10, 18,  9, 20,  1,  3,  8,  6,  1, 17, 20, 20],
        [ 3,  4,  9, 17, 10, 13, 16, 10,  6,  1, 17, 10,  5, 17,  8, 10, 10, 13,
          8,  1, 10, 11, 10, 18,  9, 17,  8,  1,  4, 10],
        [10,  8, 17, 13, 12,  7, 18, 16,  5,  8,  3,  6,  8, 10, 10,  6, 10,  5,
         10, 13, 18, 15, 13, 18,  5,  1, 18, 20, 17, 16]])
cds= torch.tensor([[65, 31, 22, 56, 56, 45, 62, 55, 31, 61, 50, 59, 10,  9, 45, 62, 50, 62,
         28, 24, 56, 33, 19,  8, 28, 13, 45, 21, 55,  6],
        [65, 55, 29, 23, 50, 62, 52, 52, 58, 55, 39, 30, 58, 52, 49, 51, 51, 43,
         51, 29, 49, 43,  7, 54, 51, 33, 56, 62, 34,  7],
        [65, 51, 59, 43, 42, 25, 30, 10,  9, 56, 58, 46,  9,  5, 38, 41, 29, 25,
         22, 33, 62, 29, 45, 13, 49, 43, 46, 33, 54, 59],
        [65, 25, 33, 46, 18, 35, 23, 53,  2,  1, 33, 51, 56, 33, 13, 17, 60, 29,
          1,  9, 18, 61, 20, 26, 61,  1, 54, 49,  7, 38]])

model.eval()

dummy_input = (prot, cds)
input_names = ("prot", "cds")
output_names = ["output"]
model.eval()
torch.onnx.export(model, dummy_input, "test.onnx", input_names=input_names, output_names=output_names,
                export_params=True, opset_version=18, verbose=True)

# with torch.profiler.profile(with_stack=True, profile_memory=True) as prof:
#     model(*dummy_input)

# print(prof.key_averages().table(sort_by="self_cpu_time_total"))
# print(torch.jit.trace(model, example_inputs=dummy_input))
# for avg in prof.key_averages():
#     print(avg.stack)
