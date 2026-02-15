DATA_FOLDER="${HOME}/chLi/Dataset/GS/haizei_1_v4"

ITERATIONS=30000

CUDA_VISIBLE_DEVICES=1 \
  python mesh_extract.py \
  -m ${DATA_FOLDER}/gggs/
