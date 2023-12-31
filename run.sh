python run.py --batch_size 64 \
          --save_model_path '/content/drive/MyDrive/Master/NomTemple/CLS/v1' \
          --dataset_root  /content/han_nom_cls_v2 \
          --epochs 100 \
          --startIter 0 \
          --maxIter 1000 \
          --valIter 30 \
          --train True \
          --test True \
          --learning_rate 1.0 \
          --rho 0.9 \
          --eps 1e-05 \
          --seed 2103 \
          --mode han_nom_cls

        #   --checkpoint_path '/content/drive/MyDrive/Master/NLP/Experiment/Protonet/5w1s/best_model.pth' \