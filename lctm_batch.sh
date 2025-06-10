source .env
bash run_lctm.sh -m Llama-3.2-3B-Instruct2 -b max
bash run_lctm.sh -m Llama-3.2-3B-Instruct2 
bash run_lctm.sh -m Llama-3.2-3B-Instruct2 -b max -h
bash run_lctm.sh -m Llama-3.2-3B-Instruct2 -h

bash run_lctm.sh -m Llama-3.2-3B -b max
bash run_lctm.sh -m Llama-3.2-1B-Instruct -b max
bash run_lctm.sh -m Llama-3.2-1B -b max
bash run_lctm.sh -m original_transformer -b max
bash run_lctm.sh -m original_transformer.glove -b max


bash run_lctm.sh -m Llama-3.2-3B -b max -h
bash run_lctm.sh -m Llama-3.2-1B-Instruct -b max -h
bash run_lctm.sh -m Llama-3.2-1B -b max -h
bash run_lctm.sh -m original_transformer -b max -h
bash run_lctm.sh -m original_transformer.glove -b max -h
