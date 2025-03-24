import subprocess

model_names = {
    'CompressMamba',
    'DeepDNA_Attn',
    'DeepDNA',
    'DeepDNA_DB',
    'CompressAttn',
}


commands = [
    "deepspeed train_model.py CompressMamba 0 > ./train_logs/train_compressmamba_single.log 2>&1",
    "deepspeed train_model.py CompressMamba 128 > ./train_logs/train_compressmamba_128.log 2>&1",
    "deepspeed train_model.py CompressMamba 256 > ./train_logs/train_compressmamba_256.log 2>&1",
    "deepspeed train_model.py CompressMamba 512 > ./train_logs/train_compressmamba_512.log 2>&1",
    "deepspeed train_model.py CompressMamba 768 > ./train_logs/train_compressmamba_768.log 2>&1",
    "deepspeed train_model.py CompressMamba 1024 > ./train_logs/train_compressmamba_1024.log 2>&1",
    "deepspeed train_model.py CompressMamba 2048 > ./train_logs/train_compressmamba_2048.log 2>&1",
    "deepspeed train_model.py CompressMamba 4096 > ./train_logs/train_compressmamba_4096.log 2>&1",
    "deepspeed train_model.py CompressMamba 8192 > ./train_logs/train_compressmamba_8192.log 2>&1",

    "deepspeed train_model.py CompressMamba 128 2 > ./train_logs/train_compressmamba_128_2.log 2>&1",
    "deepspeed train_model.py CompressMamba 128 3 > ./train_logs/train_compressmamba_128_3.log 2>&1",
    "deepspeed train_model.py CompressMamba 128 4 > ./train_logs/train_compressmamba_128_4.log 2>&1",
    "deepspeed train_model.py CompressMamba 128 5 > ./train_logs/train_compressmamba_128_5.log 2>&1",

    "deepspeed train_model.py CompressMamba 256 2 > ./train_logs/train_compressmamba_256_2.log 2>&1",
    "deepspeed train_model.py CompressMamba 256 3 > ./train_logs/train_compressmamba_256_3.log 2>&1",
    "deepspeed train_model.py CompressMamba 256 4 > ./train_logs/train_compressmamba_256_4.log 2>&1",
    "deepspeed train_model.py CompressMamba 256 5 > ./train_logs/train_compressmamba_256_5.log 2>&1",
    
    "deepspeed train_model.py CompressMamba 512 2 > ./train_logs/train_compressmamba_512_2.log 2>&1",
    "deepspeed train_model.py CompressMamba 512 3 > ./train_logs/train_compressmamba_512_3.log 2>&1",
    "deepspeed train_model.py CompressMamba 512 4 > ./train_logs/train_compressmamba_512_4.log 2>&1",
    "deepspeed train_model.py CompressMamba 512 5 > ./train_logs/train_compressmamba_512_5.log 2>&1",

    "deepspeed train_model.py CompressMamba 768 2 > ./train_logs/train_compressmamba_768_2.log 2>&1",
    "deepspeed train_model.py CompressMamba 768 3 > ./train_logs/train_compressmamba_768_3.log 2>&1",
    "deepspeed train_model.py CompressMamba 768 4 > ./train_logs/train_compressmamba_768_4.log 2>&1",
    "deepspeed train_model.py CompressMamba 768 5 > ./train_logs/train_compressmamba_768_5.log 2>&1",

    "deepspeed train_model.py CompressMamba 1024 2 > ./train_logs/train_compressmamba_1024_2.log 2>&1",
    "deepspeed train_model.py CompressMamba 1024 3 > ./train_logs/train_compressmamba_1024_3.log 2>&1",
    "deepspeed train_model.py CompressMamba 1024 4 > ./train_logs/train_compressmamba_1024_4.log 2>&1",
    "deepspeed train_model.py CompressMamba 1024 5 > ./train_logs/train_compressmamba_1024_5.log 2>&1",

    "deepspeed train_model.py CompressMamba 2048 2 > ./train_logs/train_compressmamba_2048_2.log 2>&1",
    "deepspeed train_model.py CompressMamba 2048 3 > ./train_logs/train_compressmamba_2048_3.log 2>&1",
    "deepspeed train_model.py CompressMamba 2048 4 > ./train_logs/train_compressmamba_2048_4.log 2>&1",
    "deepspeed train_model.py CompressMamba 2048 5 > ./train_logs/train_compressmamba_2048_5.log 2>&1",

    "deepspeed train_model.py DeepDNA 0 > ./train_logs/train_deepdna_single.log 2>&1",
    "deepspeed train_model.py DeepDNA_DB 0 > ./train_logs/train_deepdna_db.log 2>&1",
    "deepspeed train_model.py DeepDNA_Attn 0 > ./train_logs/train_deepdna_attn_single.log 2>&1",
    "deepspeed train_model.py CompressAttn 0 > ./train_logs/train_compress_attn_single.log 2>&1"
]


# 执行命令的函数
def run_command(command):
    try:
        # 使用 subprocess.run 执行命令
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 打印命令的输出
        print(f"命令 '{command}' 执行成功。输出：\n{result.stdout.decode()}")
        return True
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，打印错误信息并退出
        print(f"命令 '{command}' 执行失败。错误信息：\n{e.stderr.decode()}")
        return False
        

# 依次执行每个命令
for command in commands:
    print(f"正在执行命令：{command}...")
    if not run_command(command):
        print("命令执行失败，退出。")

print("所有命令执行完毕。")