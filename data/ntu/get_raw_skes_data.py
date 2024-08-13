
import os.path as osp
import os
import numpy as np
import pickle
import logging


def get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger):
    """
    Get raw bodies data from a skeleton sequence.

    Each body's data is a dict that contains the following keys:
      - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
      - colors: raw 2D color locations. Shape: (num_frames, 25, 2)
      - interval: a list which stores the frame indices of this body.
      - motion: motion amount (only for the sequence with 2 or more bodyIDs).

    skes_path: 骨架文件所在的目录路径。
    ske_name: 骨架文件的名称（不含扩展名）。
    frames_drop_skes: 用于记录丢失帧的字典。
    frames_drop_logger: 用于记录丢失帧信息的日志记录器。

    Return:
      a dict for a skeleton sequence with 3 key-value pairs:
        - name: the skeleton filename.
        - data: a dict which stores raw data of each body.
        - num_frames: the number of valid frames.
    """
    ske_file = osp.join(skes_path, ske_name + '.skeleton') # 一个skeleton file的path
    if not osp.exists(ske_file):
        'Error: Skeleton file %s not found' % ske_file
    # Read all data from .skeleton file into a list (in string format)
    print('Reading data from %s' % ske_file[-29:])
    with open(ske_file, 'r') as fr:
        str_data = fr.readlines() # 将文件的每一行读入列表str_data中

    num_frames = int(str_data[0].strip('\r\n'))
    frames_drop = []
    bodies_data = dict()
    valid_frames = -1  # 0-based index
    current_line = 1 #

    # 遍历每一帧：
    for f in range(num_frames):
        num_bodies = int(str_data[current_line].strip('\r\n'))
        current_line += 1 # str_data[2]是body key_info

        # 处理无数据帧：
        if num_bodies == 0:  # no data in this frame, drop it
            frames_drop.append(f)  # 0-based index list of 丢失帧的帧号
            continue
        # 初始化存储关节和颜色数据的数组：
        valid_frames += 1 # 从0开始
        joints = np.zeros((num_bodies, 25, 3), dtype=np.float32) # b*25*3 三维
        colors = np.zeros((num_bodies, 25, 2), dtype=np.float32) # b*25*2 三维
        # 遍历每个身体：
        for b in range(num_bodies):
            bodyID = str_data[current_line].strip('\r\n').split()[0] # 提取bodyID
            current_line += 1 # joint数：25
            num_joints = int(str_data[current_line].strip('\r\n'))  # 25 joints
            current_line += 1 # 第一帧第一行

            for j in range(num_joints):
                temp_str = str_data[current_line].strip('\r\n').split()
                joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32) # 取前三个xyz locations, b代表第b个body，j代表第j个关节
                colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32) # 2d colors
                current_line += 1

            if bodyID not in bodies_data:  # Add a new body's data in the dict
                body_data = dict()
                body_data['joints'] = joints[b]  # ndarray: (25, 3) ！！取此帧里一个body的所有关节的location数据
                body_data['colors'] = colors[b, np.newaxis]  # ndarray: (1, 25, 2)
                body_data['interval'] = [valid_frames]  # the index of the first frame
                bodies_data[bodyID] = body_data  # Update bodies_data
            else:  # Update an already existed body's data
                body_data = bodies_data[bodyID]
                # Stack each body's data of each frame along the frame order
                body_data['joints'] = np.vstack((body_data['joints'], joints[b])) # 比如第2帧：（2，25，3）
                body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis])) # 比如第2帧：（2，25，2）
                pre_frame_idx = body_data['interval'][-1] # 获取当前身体数据中最后一个已记录帧的索引。
                body_data['interval'].append(pre_frame_idx + 1)  # add a new frame index


    num_frames_drop = len(frames_drop)
    if not (num_frames_drop < num_frames):
        print('Error: All frames data (%d) of %s is missing or lost' % (num_frames, ske_name))
    # 这里应该需要处理。。

    if num_frames_drop > 0: # 即此文件中有丢失帧， 把文件名和丢失的帧数全部写入frame.drop.log中
        frames_drop_skes[ske_name] = np.array(frames_drop, dtype=int)
        frames_drop_logger.info('{}: {} frames missed: {}\n'.format(ske_name, num_frames_drop,
                                                                    frames_drop))

    # Calculate motion (only for the sequence with 2 or more bodyIDs)
    if len(bodies_data) > 1:
        for body_data in bodies_data.values():
            body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0))
            # 求每个关节在 x, y, z 方向上的方差。
            # 然后将所有关节在 x, y, z 方向上的方差加总，得到一个标量值。
            # 这个标量值被认为是该身体的“运动量”，因为它反映了关节位置随时间的变化程度。

    return {'name': ske_name, 'data': bodies_data, 'num_frames': num_frames - num_frames_drop}


def get_raw_skes_data():
    # # save_path = './data'
    # # skes_path = '/data/pengfei/NTU/nturgb+d_skeletons/'
    # stat_path = osp.join(save_path, 'statistics')
    #
    # skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    # save_data_pkl = osp.join(save_path, 'raw_skes_data.pkl')
    # frames_drop_pkl = osp.join(save_path, 'frames_drop_skes.pkl')
    #
    # frames_drop_logger = logging.getLogger('frames_drop')
    # frames_drop_logger.setLevel(logging.INFO)
    # frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'frames_drop.log')))
    # frames_drop_skes = dict()

    skes_name = np.loadtxt(skes_name_file, dtype=str) # 一维

    num_files = skes_name.size # 文件的数量
    print('Found %d available skeleton files.' % num_files)

    raw_skes_data = [] # 一个列表，用于存储每个骨架文件的处理结果。
    frames_cnt = np.zeros(num_files, dtype=int) # frames_cnt 是一个数组，用于记录每个文件的有效帧数。初始为0.

    # 遍历每一个file：
    for (idx, ske_name) in enumerate(skes_name):
        bodies_data = get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger)
        raw_skes_data.append(bodies_data)
        frames_cnt[idx] = bodies_data['num_frames'] # 赋值
        # 为了避免处理过程中有错误发生，这里把raw_skes_data这个list中的每个字典元素保存到文件里??

        if (idx + 1) % 1000 == 0: # idx 也是从0开始的，每处理1000个文件，打印进度信息。
            print('Processed: %.2f%% (%d / %d)' % \
                  (100.0 * (idx + 1) / num_files, idx + 1, num_files))

    # 保存处理后的数据：
    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
    np.savetxt(osp.join(save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d') # np.savetxt 保存每个文件的帧计数信息。

    print('Saved raw bodies data into %s' % save_data_pkl)
    print('Total frames: %d' % np.sum(frames_cnt))

    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    save_path = '/content/drive/Othercomputers/我的笔记本电脑/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition/data/ntu'

    skes_path = r"/content/drive/Othercomputers/我的笔记本电脑/nturgb+d_skeletons" # Dataset path
    stat_path = osp.join(save_path, 'statistics')
    if not osp.exists('./raw_data'):
        os.makedirs('./raw_data')

    skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    save_data_pkl = osp.join(save_path, 'raw_data', 'raw_skes_data.pkl')
    frames_drop_pkl = osp.join(save_path, 'raw_data', 'frames_drop_skes.pkl') # 数据字典序列化并存入这个地址

    frames_drop_logger = logging.getLogger('frames_drop')  # 创建一个名为 'frames_drop' 的日志记录器（logger）。
    frames_drop_logger.setLevel(logging.INFO) # 设置日志记录器的级别为 INFO
    frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'raw_data', 'frames_drop.log')))
    # 日志记录器添加一个文件处理器（FileHandler）,用于将日志输出到指定的文件中。
    # ./raw_data/frames_drop.log

    # 创建一个空的字典对象，命名为 frames_drop_skes。
    frames_drop_skes = dict()

    get_raw_skes_data()

    # 'wb' 模式表示以二进制写模式打开文件。这是因为pickle在处理二进制数据时需要使用二进制模式。
    # pickle.dump(obj, file, protocol) 函数用于将对象obj序列化，并将其写入到file中。
    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)

