python feature_extract_ib.py -config train_AWIBF_0.001_RADAM.yaml
python feature_extract_ib.py -config train_0.001_RADAM.yaml
python feature_extract_ib.py -config extract_AWIBF_0.001_RADAM_TF=True.yaml
python feature_extract_ib.py -config extract_AWIBF_0.001_RADAM_TF=False.yaml
python feature_extract_ib.py -config extract_0.001_RADAM_TF=True.yaml
python feature_extract_ib.py -config extract_0.001_RADAM_TF=False.yaml
cd ../ASFormer
python main.py --action train --name 5fps_TF=False_extract_AWIBF_0.001_RADAM --features_dim 2048
python main.py --action train --name 5fps_TF=True_extract_AWIBF_0.001_RADAM --features_dim 2055
python main.py --action train --name 5fps_TF=True_extract_0.001_RADAM --features_dim 2055
python main.py --action train --name 5fps_TF=False_extract_0.001_RADAM --features_dim 2048
python main.py --action predict --name 5fps_TF=False_extract_AWIBF_0.001_RADAM --features_dim 2048 --epoch 30 
python main.py --action predict --name 5fps_TF=True_extract_AWIBF_0.001_RADAM --features_dim 2055 --epoch 30
python main.py --action predict --name 5fps_TF=True_extract_0.001_RADAM --features_dim 2055 --epoch 30
python main.py --action predict --name 5fps_TF=False_extract_0.001_RADAM --features_dim 2048 --epoch 30

