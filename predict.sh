echo Predict
python predict.py --dataset_root /content/han_nom_kinh_ky \
                  --save_model_path /content/drive/MyDrive/Master/NomTemple/CLS/v2 \
                  --checkpoint_path /content/drive/MyDrive/Master/NomTemple/CLS/v2/best_accuracy.pth \
                  --mode han_nom_cls \
                  --batch_size 64 \
                  --num_class 4

echo zip file