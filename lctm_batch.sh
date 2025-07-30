source .env
# bash run_lctm.sh -m Llama-3.2-3B-Instruct2 -b max -d epoch
# bash run_lctm.sh -m Llama-3.2-3B-Instruct2 
# bash run_lctm.sh -m Llama-3.2-3B-Instruct2 -b max -h
# bash run_lctm.sh -m Llama-3.2-3B-Instruct2 -h

# bash run_lctm.sh -m Llama-3.2-3B -b max
# bash run_lctm.sh -m Llama-3.2-1B-Instruct -b max
# bash run_lctm.sh -m Llama-3.2-1B -b max
# bash run_lctm.sh -m original_transformer -b max
# bash run_lctm.sh -m original_transformer.glove -b max


# bash run_lctm.sh -m Llama-3.2-3B -b max -h
# bash run_lctm.sh -m Llama-3.2-1B-Instruct -b max -h
# bash run_lctm.sh -m Llama-3.2-1B -b max -h
# bash run_lctm.sh -m original_transformer -b max -h
# bash run_lctm.sh -m original_transformer.glove -b max -h

bash run_lctm.sh -m Llama-3.2-3B-Instruct2 -b max -d epoch
bash run_lctm.sh -m Llama-3.2-3B-Instruct2 -b default -d epoch

bash run_lctm.sh -m Llama-3.2-3B -b max -d epoch
bash run_lctm.sh -m Llama-3.2-3B -b default -d epoch

bash run_lctm.sh -m Llama-3.2-1B-Instruct -b max -d epoch
bash run_lctm.sh -m Llama-3.2-1B-Instruct -b default -d epoch

bash run_lctm.sh -m Llama-3.2-1B -b max -d epoch
bash run_lctm.sh -m Llama-3.2-1B -b default -d epoch

bash run_lctm.sh -m original_transformer -b max -d epoch
bash run_lctm.sh -m original_transformer -b default -d epoch

bash run_lctm.sh -m original_transformer.glove -b max -d epoch
bash run_lctm.sh -m original_transformer.glove -b default -d epoch

bash run_lctm.sh -m Llama-3.2-3B-2 -b max -d epoch
bash run_lctm.sh -m Llama-3.2-3B-2 -b default -d epoch

bash run_lctm.sh -m Llama-3.2-3B-2 -b max -d accumulation
bash run_lctm.sh -m Llama-3.2-3B-2 -b default -d accumulation