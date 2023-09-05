echo Predict
python predict.py --dataset_root /content/han_nom_kinh_ky \
                  --save_model_path /content/drive/MyDrive/Master/NomTemple/CLS/v1 \
                  --checkpoint_path /content/drive/MyDrive/Master/NomTemple/CLS/v1/best_accuracy.pth \
                  --mode han_nom_cls \
                  --batch_size 1024

echo zip file