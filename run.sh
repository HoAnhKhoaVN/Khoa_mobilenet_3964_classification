python run.py --batch_size 8192 \
          --save_model_path '/content/drive/MyDrive/Master/NLP/Experiment/Protonet/5w1s' \
          --checkpoint_path '/content/drive/MyDrive/Master/NLP/Experiment/Protonet/5w1s/best_model.pth' \
          --dataset_root  /content/dataset_han_nom_real_synth \
          --epochs 100 \
          --startIter 0 \
          --maxIter 100000 \
          --valIter 100 \
          --train \
          --test \
          --learning_rate 1.0 \
          --rho 0.9 \
          --eps 1e-05 \
          --seed 2103 \
          --mode maml
