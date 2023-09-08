import os
from tqdm import tqdm
if __name__ == '__main__':
    # region Parameters
    LOG_PREDICT_PATH = '/content/test_file.txt'
    ROOT_DIR = '/content/cls_v1'
    MAX_CONFIDENT = 0.999
    # endregion

    # region Read file
    print("Read file")
    with open( LOG_PREDICT_PATH, 'r', encoding = 'utf-8') as file:
        data = file.readlines()
    data = list(map(lambda x: x.strip(), data))
    print(f'==> Length of file: {len(data)}')
    # endregion

    # region Create dictionary
    print("Create dictionary")
    name_class = ['giay', 'go', 'vat_lieu_khac', 'xi_mang']
    str_num = list(map(lambda x: str(x), range(len(name_class))))
    num2class = dict(zip( str_num, name_class))
    print(f'==> num2class: {num2class}')
    # endregion

    # region Create necessary folders
    print("Create necessary folders")
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR, exist_ok= True)

    for name_fd in name_class:
        fd_path = os.path.join(ROOT_DIR, name_fd)
        print(f'==> Create {fd_path}')
        os.makedirs(
            name = fd_path,
            exist_ok= True
        )
        for sub_fd in ['certainty', 'distrust']:
            sub_fd_path = os.path.join(fd_path, sub_fd)
            print(f'====> Create {sub_fd_path}')
            os.makedirs(
                name = sub_fd_path,
                exist_ok= True
            )
    # endregion

    # region Move file
    for line in tqdm(data):
        src_path, class_index, conf = line.split('\t')
        class_name = num2class[class_index.strip()]
        file_name = src_path.split('/')[-1]
        print(f'File name : {file_name}')
        print(f'src_path : {src_path} - type: {type(src_path)}')
        print(f'class_index: {class_name} - type: {type(class_index)}')
        print(f'conf: {conf} - type: {type(conf)}')

        if float(conf) > MAX_CONFIDENT: # Certainty
            tgt_path = os.path.join(ROOT_DIR, class_name, 'certainty', file_name)
        else:
            tgt_path = os.path.join(ROOT_DIR, class_name, 'distrust', file_name)

        # Move file        
        os.system(
            command= f'mv {src_path} {tgt_path}'
        )
    # endregion
