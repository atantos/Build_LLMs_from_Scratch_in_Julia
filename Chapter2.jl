#  2.2 Tokenizing Text 

using HTTP
using Base.Threads

function download_github_raw(url::String)
    response = HTTP.get(url)
    return String(response.body)
end

url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

raw_text = download_github_raw(url)

# for demonstation purposes, we will only show the first 100 characters of the text.
println("Total number of character:", length(raw_text))
println(raw_text[begin:99])

# Before transitioning to a prebuilt tokenizer

# import re
# text = "Hello, world. This, is a test."
# result = re.split(r'(\s)', text)
# print(result)

text = "Hello, world. This, is a test."
# result = split(text, r"\s") does not retain the whitespaces as with the split() in python.
# Capturing groups do not affect the behavior of split(). So, the whitespaces are not retained as with the split() in python.
# result = split(text, r"(\s)") <=> result = split(text, r"\s")

# Benchmark regex matching with retention of whitespaces
using BenchmarkTools

# Benchmark with the custom options
@btime result = [m.match for m in eachmatch(r"\S+|\s", text)] evals = 1000

println(result)

# ==============================

# result = re.split(r'([,.]|\s)', text)
# print(result)

result = split(text, r"([,.]|\s)")
println(result)


# result = [item for item in result if item.strip()]
# print(result)  
# Note on the difference between list/array comprehension in Python and Julia
filter!(x -> x != "", result)
println(result)

# "Let’s modify it a bit further so that it can also handle other types of punctuation, such as question marks, quotation marks, and the double-dashes"
# text = "Hello, world. Is this-- a test?"
# result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# result = [item.strip() for item in result if item.strip()]
# print(result)

text = """Hello, world. Is this-- a test?
    new
    line and another one. And
"""

# Define the regular expression pattern
pattern = r"([,.:;?_!\"()\']|--|\s)"

# Function to split text by matches while keeping the delimiters
import Base.split
function split(pattern, text, with_dlm=true)
    result = String[]
    last_index = 1

    for m in eachmatch(pattern, text)
        start = first(m.offsets) # Get the start index of the match
        stop = last(m.offsets) # Get the end index of the match

        # Get the substring between matches
        part = last_index < start && length(m.match) > 1 ? text[last_index:start-1] : strip(text[last_index:start-1], '-')

        # Only push non-empty parts
        if part != ""
            push!(result, part)
        end

        # leave out unnecessary empty strings 
        if with_dlm && m.match != ""
            push!(result, m.match)
        end
        last_index = stop + 1
    end

    # Add the final substring if it's not empty
    if last_index <= length(text)
        final_part = text[last_index:end]
        if final_part != ""
            push!(result, final_part)
        end
    end

    return result
end

# Define the regular expression pattern
pattern = r"([,.:;?_!\"()']|--|\s)"

# Get the result
result = split(pattern, text, false)

result = strip.(result)

# Remove empty strings from the result
result = filter(x -> !(x in ([" ", "\n"])), text)


println(result)

# Getting back to the original text


# preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
# preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed))


preprocessed = split(r"([,.:;?_!\"()']|--|\s)", raw_text)
# 3 ways of achieving the same result
# with array comprehension (4694 in Python we get 4690)
preprocessed = [strip(item) for item in preprocessed if strip(item) != ""]

println(length(preprocessed))
println(preprocessed)

println(preprocessed[begin:30])

# 2.3 Converting tokens into IDs

# all_words = sorted(set(preprocessed))
# vocab_size = len(all_words)
# print(vocab_size)

all_words = sort(collect(Set(preprocessed)))
vocab_size = length(all_words)
println(vocab_size)

# Listing 2.2 Creating a vocabulary     
# vocab = {token:integer for integer,token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break  

vocab = Dict(word => idx for (idx, word) in enumerate(all_words))
for (i, item) in enumerate(vocab)
    println(item)
    if i >= 50
        break
    end
end

# class SimpleTokenizerV1:
#     def __init__(self, vocab):
#         self.str_to_int = vocab #A
#         self.int_to_str = {i:s for s,i in vocab.items()} #B

#     def encode(self, text): #C
#         preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
#         preprocessed = [
#             item.strip() for item in preprocessed if item.strip()
#         ]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids

#     def decode(self, ids): #D
#         text = " ".join([self.int_to_str[i] for i in ids]) 

#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #E
#         return text


# helper function 
function prealloc_str_int(preprocessed, vocab)
    n = length(preprocessed)
    result = Vector{Int}(undef, n)
    i = 1
    for s in preprocessed
        result[i] = vocab[s]
        i += 1
    end
    return result
end


function prealloc_int_str(vocab, ids)
    n = length(ids)
    result = Vector{String}(undef, n)
    i = 1
    for s in ids
        println(vocab[s])
        result[i] = vocab[s]
        i += 1
    end
    return result
end

struct SimpleTokenizerV1
    str_to_int::Dict{String,Int}
    int_to_str::Dict{Int,String}

    function SimpleTokenizerV1(vocab::Dict{<:AbstractString,Int})
        str_to_int = vocab
        int_to_str = Dict(i => s for (s, i) in vocab)
        new(str_to_int, int_to_str)
    end
end

function encode(tokenizer::SimpleTokenizerV1, text::String)
    preprocessed = split(r"([,.?_!\"()\']|--|\s)", text)
    preprocessed = filter(x -> !(x in ([" ", "\n"])), preprocessed)
    # join([tokenizer.int_to_str[i] for i in ids], " ")
    ids = prealloc_str_int(preprocessed, tokenizer.str_to_int)
    return ids
end

function decode(tokenizer::SimpleTokenizerV1, ids::Vector{Int})
    text = join(prealloc_int_str(tokenizer.int_to_str, ids), " ")
    text = replace(text, r"\s+([,.?!\"()\'])" => s"\1")
    return text
end

# Example usage
vocab = Dict("hello" => 1, "world" => 2, "," => 3, "!" => 4)
tokenizer = SimpleTokenizerV1(vocab)

# Encoding
text = "hello, world!"
encoded = encode(tokenizer, text)
println(encoded)  # Output: [1, 3, 2, 4]

# Decoding
decoded = decode(tokenizer, encoded)
println(decoded)  # Output: "hello, world!"

# 2.4 Adding special context tokens

# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# vocab = {token:integer for integer,token in enumerate(all_tokens)}

all_tokens = sort(collect(Set(preprocessed)))
push!(all_tokens, "<|endoftext|>", "<|unk|>")
vocab = Dict(token => integer for (integer, token) in enumerate(all_tokens))

# check whether the special tokens are in the vocab
# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)

# we don't need to use the for loop to print the last 5 items
all(key -> haskey(vocab, key), ["<|endoftext|>", "<|unk|>"])


# class SimpleTokenizerV2:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = { i:s for s,i in vocab.items()}

#     def encode(self, text):
#         preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
#         preprocessed = [
#             item.strip() for item in preprocessed if item.strip()
#         ]
#         preprocessed = [item if item in self.str_to_int  #A
#                         else "<|unk|>" for item in preprocessed]

#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids

#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])

#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #B
#         return text

struct SimpleTokenizerV2
    str_to_int::Dict{String,Int}
    int_to_str::Dict{Int,String}

    function SimpleTokenizerV2(vocab::Dict{<:AbstractString,Int})
        str_to_int = vocab
        int_to_str = Dict(i => s for (s, i) in vocab)
        new(str_to_int, int_to_str)
    end
end

function encode(tokenizer::SimpleTokenizerV2, text::String)
    preprocessed = split(r"([,.?_!\"()\']|--|\s)", text)
    preprocessed = filter(x -> !(x in ([" ", "\n"])), preprocessed)
    println(preprocessed)
    preprocessed = [haskey(tokenizer.str_to_int, item) ? item : "<|unk|>" for item in preprocessed]
    ids = prealloc_str_int(preprocessed, tokenizer.str_to_int)
    return ids
end

function decode(tokenizer::SimpleTokenizerV2, ids::Vector{Int})
    text = join(prealloc_int_str(tokenizer.int_to_str, ids), " ")
    text = replace(text, r"\s+([,.?!\"()\'])" => s"\1")
    return text
end


# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))
# print(text)

text1 = "Hello, do you like, tea?"
text2 = "In the sunlit terraces of the palace."
text = join([text1, text2], " <|endoftext|> ")

tokenizer2 = SimpleTokenizerV2(vocab)

# Encoding
encoded = encode(tokenizer2, text)
println(encoded)  # Output: [1, 3, 2, 4]

# Decoding
decoded = decode(tokenizer2, encoded)
println(decoded)  # Output: "hello, world!"


# 2.5 Byte Pair Encoding (BPE)

# from importlib.metadata import version
# import tiktoken
# print("tiktoken version:", version("tiktoken"))

using BytePairEncoding

# text = (
#     "Hello, do you like tea? <|endoftext|> In the sunlit terraces",
#     "of someunknownPlace."
# )
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)

# tokenizer = tiktoken.get_encoding("gpt2")
# Thorough discussion on gpt-2 and tiktoken (which is the OpenAI's BPE encoding format) https://chatgpt.com/share/67248e53-89ac-8008-b643-4d0accce91e8

# tokenizer = BytePairEncoding.load_tiktoken("gpt2")
tokenizer = BytePairEncoding.load_tiktoken_encoder("gpt2")
integers = tokenizer.encode("Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.")
println(integers)

# strings = tokenizer.decode(integers)
# print(strings)

strings = tokenizer.decode(integers)
println(strings)

# 2.6 Data sampling with a sliding window

# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))

enc_text = tokenizer.encode(raw_text)
print(length(enc_text))

# enc_sample = enc_text[50:] starting from 51 due to python's zero indexing

enc_sample = enc_text[51:end]

# context_size = 4         
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]
# print(f"x: {x}")
# print(f"y:      {y}") 

context_size = 4
x = enc_sample[1:context_size]
y = enc_sample[2:context_size+1]
print("x: $x")
print("y:      $y")

# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(context, "---->", desired)

# Preallocate context as an empty array with a maximum possible length
context = Vector{eltype(enc_sample)}()

for i in 1:context_size
    # Add the current element to the context
    push!(context, enc_sample[i])
    desired = enc_sample[i+1]
    println(context, " ----> ", desired)
end

# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

for i in 1:context_size
    # Add the current element to the context
    push!(context, enc_sample[i])
    desired = enc_sample[i+1]
    println(tokenizer.decode(context), " ---->", tokenizer.decode([desired]))
end

# import torch
# from torch.utils.data import Dataset, DataLoader
# class GPTDatasetV1(Dataset):
#     def __init__(self, txt, tokenizer, max_length, stride):
#         self.input_ids = []
#         self.target_ids = []

#         token_ids = tokenizer.encode(txt)    #1

#         for i in range(0, len(token_ids) - max_length, stride):     #2
#             input_chunk = token_ids[i:i + max_length]
#             target_chunk = token_ids[i + 1: i + max_length + 1]
#             self.input_ids.append(torch.tensor(input_chunk))
#             self.target_ids.append(torch.tensor(target_chunk))

#     def __len__(self):    #3
#         return len(self.input_ids)

#     def __getitem__(self, idx):         #4
#         return self.input_ids[idx], self.target_ids[idx]

using Flux
# The MLUtils package allow for creating a dataloader compatible with Flux.jl
using MLUtils
using BytePairEncoding

# using Transformers I can use the BytePairEncoding tokenizer from the BytePairEncodings.jl package and skip the Transformers.jl package

# Define custom dataset
struct GPTDatasetV1
    input_ids::Vector{Vector{Int}}
    target_ids::Vector{Vector{Int}}
end

function GPTDatasetV1(txt, tokenizer; max_length, stride)
    input_ids = Vector{Vector{Int}}()
    target_ids = Vector{Vector{Int}}()

    # Tokenize the input text
    token_ids = tokenizer.encode(txt)

    # Generate input and target chunks
    for i in 1:stride:(length(token_ids)-max_length)
        input_chunk = token_ids[i:i+max_length-1]
        target_chunk = token_ids[i+1:i+max_length]
        push!(input_ids, input_chunk)
        push!(target_ids, target_chunk)
    end

    return GPTDatasetV1(input_ids, target_ids)
end

# Define the length function
Base.length(dataset::GPTDatasetV1) = length(dataset.input_ids)

# Define the indexing function
function Base.getindex(dataset::GPTDatasetV1, idx)
    return dataset.input_ids[idx], dataset.target_ids[idx]
end

# Example usage:

tokenizer = BytePairEncoding.load_tiktoken_encoder("gpt2")
dataset = GPTDatasetV1(raw_text, tokenizer, max_length=128, stride=64)
data_loader = DataLoader(dataset, batchsize=16, shuffle=true, partial=false, parallel=false)

# def create_dataloader_v1(txt, batch_size=4, max_length=256,
#                          stride=128, shuffle=True, drop_last=True,
#                          num_workers=0):
#     tokenizer = tiktoken.get_encoding("gpt2")                         #1
#     dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)   #2
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         drop_last=drop_last,     #3
#         num_workers=num_workers     #4
#     )

#     return dataloader

function create_dataloader_v1(txt; batch_size=4, max_length=6,
    stride=4, shuffle=true, partial=false, parallel=false)
    tokenizer = BytePairEncoding.load_tiktoken_encoder("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length=max_length, stride=stride)

    return DataLoader(dataset, batchsize=batch_size, shuffle=shuffle, partial=partial, parallel=parallel)
end

raw_text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi turpis diam, sollicitudin non pellentesque quis, lobortis tempor augue. Duis vestibulum id mi ut pellentesque. In hac habitasse platea dictumst. Praesent varius lectus neque, non mattis tortor vulputate ut. Fusce rhoncus elementum sodales. Quisque vel pellentesque leo. Quisque dapibus aliquam volutpat. Sed nec arcu sollicitudin, placerat mauris sed, pulvinar metus. Quisque volutpat est elit, vel tincidunt nibh congue quis. Vivamus eleifend luctus diam vel placerat. Praesent in velit mauris. Nulla non diam ac est mattis sagittis eget eu mauris.
Aliquam id mi vulputate, mattis erat in, sodales purus. Vivamus at purus eget ex commodo tristique ut ac magna. Cras efficitur dictum dui, eu efficitur ex aliquet eu. Donec vulputate odio eu sapien pharetra, vel mattis eros tristique. Nam vehicula mi non porta commodo. Pellentesque suscipit, sem nec ultricies accumsan, nulla enim eleifend lectus, eu malesuada diam neque et metus. Pellentesque auctor ultricies viverra. Quisque ut convallis est. Etiam mollis, odio eu molestie pellentesque, nulla libero feugiat ligula, sit amet lobortis est ante in justo. Aliquam erat volutpat. Donec sagittis gravida est. Sed rutrum sem a commodo convallis. Vestibulum non convallis nisi. Sed dignissim metus eget lacus viverra ultrices. Sed et dolor ut neque mollis posuere. Quisque ac molestie leo.
Phasellus porttitor hendrerit mauris a hendrerit. Ut eget rutrum ex. Nam fermentum semper ex, eu dignissim erat suscipit eu. Curabitur condimentum efficitur tincidunt. Nullam lacus ante, faucibus vitae feugiat vel, elementum sit amet dolor. Nulla facilisi. Nam laoreet, mauris sodales ornare pretium, enim est vehicula diam, at semper nisl tortor ut dolor. Proin ullamcorper ex at aliquam pulvinar. Aenean congue aliquam sagittis. Proin efficitur condimentum sapien at pretium. Etiam laoreet, lectus nec faucibus cursus, nisi nulla fermentum metus, sed aliquet ante lacus at turpis. Etiam nec arcu est.
"""

dataloaderV0 = create_dataloader_v1(raw_text)


# Iterating over data_loader
for (input_chunk, target_chunk) in dataloaderV0
    println("Input: ========================", input_chunk)
    println("Target: ========================", target_chunk)
end

# 2.7 Creating Token Embeddings

using Random
using Flux

# token ids and defining an embedding layer
# input_ids = torch.tensor([2, 3, 5, 1])
# vocab_size = 6
# output_dim = 3  

# token ids and defining an embedding layer
input_ids = [2, 3, 5, 1]
vocab_size = 30   # replace with your vocabulary size
output_dim = 20   # replace with your desired output dimension

# Set random seed
# torch.manual_seed(123)
# embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Print the embedding weights
# print(embedding_layer.weight)  

# Get the embedding for a specific token (0-based indexing)
# print(embedding_layer(torch.tensor([3])))

# Set random seed
Random.seed!(123)
embedding_layer = Flux.Embedding(vocab_size, output_dim)

# Print the embedding weights
println(embedding_layer.weight)

# Get the embedding for a specific token 
embedding_layer.weight[1:end, 3]

# 2.8 Encoding word positions

using Flux
using Random


# Define the vocabulary size and output dimension
# vocab_size = 50257
# output_dim = 256

vocab_size = 50257
output_dim = 256

# Token embedding layer
# token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Token embedding layer
token_embedding_layer = Flux.Embedding(vocab_size, output_dim)

# Maximum sequence length
# max_length = 4

# Maximum sequence length
max_length = 4

# Example data (replace `raw_text` with actual input text)

# dataloader = create_dataloader_v1(
#     raw_text, batch_size=8, max_length=max_length,
#    stride=max_length, shuffle=False
# )
batch_size = 8
dataloader = create_dataloader_v1(raw_text, batch_size=batch_size, max_length=max_length, stride=max_length, shuffle=false)

# Simulate loading the data
# data_iter = iter(dataloader)
# inputs, targets = next(data_iter)
# print("Token IDs:\n", inputs)
# print("\nInputs shape:\\n", inputs.shape)

# Simulate loading the data
inputs, _ = first(dataloader)
println("Token IDs:\n", inputs)
println("Token IDs:\n", targets)


# Generate token embeddings
# token_embeddings = token_embedding_layer(inputs)
# print(token_embeddings.shape)
# torch.Size([8, 4, 256])

# Generate token embeddings
reduced_inputs = reduce(vcat, inputs')
token_embeddings = token_embedding_layer(reduced_inputs)
token_embeddings = permutedims(token_embeddings, (2, 3, 1))

println(size(token_embeddings))  # Output: (256, 8, 4)

# context_length = max_length
# pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# print(pos_embeddings.shape)

# Positional embedding layer
pos_embedding_layer = Flux.Embedding(max_length, output_dim)
pos_embeddings = pos_embedding_layer(collect(1:max_length))
println(size(pos_embeddings))

pos_embeddings = reshape(pos_embeddings, 1, 4, 256)  # Shape: 

pos_embeddings = repeat(pos_embeddings, 1, 8, 1) # Resulting shape: (8, 4, 256)



# Add token and positional embeddings

# input_embeddings = token_embeddings + pos_embeddings
# print(input_embeddings.shape)
# torch.Size([8, 4, 256])

input_embeddings = token_embeddings .+ pos_embeddings  # Resulting shape: (8, 4, 256)

println(size(input_embeddings))  # Output: (8, 4, 256)








