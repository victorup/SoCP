import torch
import logging

def collate_fn(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch: batch中是MyDataset中__getitem__返回的元素，形式为tuple。for us是tuple len为6的输入源，
    :return:
    """
    pad_id = 0
    x_ids = []
    x_ids_len = []
    con_ids = []
    con_ids_len = []
    char_ids = []
    p_score = []
    m_score = []
    r_score = []
    y_ids = []
    y_ids_len = []
    pmr_score = []
    batch_size = len(batch)
    max_x_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    max_con_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    max_y_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    # 计算该batch中input的最大长度
    for b_idx in range(batch_size):
        if max_x_len < len(batch[b_idx][0]):
            max_x_len = len(batch[b_idx][0])
        if max_con_len < len(batch[b_idx][2]):
            max_con_len = len(batch[b_idx][2])
        if max_y_len < len(batch[b_idx][-2]):
            max_y_len = len(batch[b_idx][-2])
    # 使用pad_id对小于max_input_len的input_id进行补全
    for b_idx in range(batch_size):
        # x
        x_len = len(batch[b_idx][0])
        x_ids.append(batch[b_idx][0])
        x_ids_len.append(batch[b_idx][1])  # x_len
        x_ids[b_idx].extend([pad_id] * (max_x_len - x_len))
        # context
        con_len = len(batch[b_idx][2])
        con_ids.append(batch[b_idx][2])
        con_ids_len.append(batch[b_idx][3])
        con_ids[b_idx].extend([pad_id] * (max_con_len - con_len))
        # char
        char_ids.append(batch[b_idx][4])
        # pmr
        p_score.append(batch[b_idx][5])
        m_score.append(batch[b_idx][6])
        r_score.append(batch[b_idx][7])
        # y
        y_len = len(batch[b_idx][8])
        y_ids.append(batch[b_idx][8])
        y_ids_len.append(batch[b_idx][9])  # y_len
        y_ids[b_idx].extend([pad_id] * (max_y_len - y_len))

    return torch.tensor(x_ids).long(), torch.tensor(x_ids_len).long(), torch.tensor(con_ids).long(), \
           torch.tensor(con_ids_len).long(), torch.tensor(char_ids).long(), torch.tensor(p_score).float(), \
           torch.tensor(m_score).float(), torch.tensor(r_score).float(), \
           torch.tensor(y_ids).long(), torch.tensor(y_ids_len).long()


def collate_fn_pmrclf(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch: batch中是MyDataset中__getitem__返回的元素，形式为tuple。for us是tuple len为6的输入源，
    :return:
    """
    pad_id = 0
    x_ids = []
    x_ids_len = []
    y = []
    batch_size = len(batch)
    max_x_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    # 计算该batch中input的最大长度
    for b_idx in range(batch_size):
        if max_x_len < len(batch[b_idx][0]):
            max_x_len = len(batch[b_idx][0])
    # 使用pad_id对小于max_input_len的input_id进行补全
    for b_idx in range(batch_size):
        # x
        x_len = len(batch[b_idx][0])
        x_ids.append(batch[b_idx][0])
        x_ids_len.append(batch[b_idx][1])  # x_len
        x_ids[b_idx].extend([pad_id] * (max_x_len - x_len))
        y.append(batch[b_idx][2])

    return torch.tensor(x_ids).long(), torch.tensor(x_ids_len).long(), torch.tensor(y).long()


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger