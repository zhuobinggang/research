# 为了“考察”部分做实验
import mainichi_paragraph as custom_data

def run():
    manual_exp_ld = custom_data.load_customized_loader(file_name='manual_exp', half=2, batch=1, mini=False, shuffle = True)
    mld = manual_exp_ld
    mld = ld_without_opening(mld)

def ld_without_opening(ld):
    ld = [case for case in ld if case[0][2] != 0]
    return ld


# 让机器来跑特定
