import torch

###############################################

EXP_NAME = "hwt_kor_lr0.00005"; RESUME = False

DATASET = 'hwt_kor'
if DATASET == 'hwt_kor':
    DATASET_PATHS = 'files_kr/korean_handwriting.pkl'
    NUM_WRITERS = 141

ENGLISH_WORDS_PATH = 'files_kr/korean.txt'

###############################################

IMG_HEIGHT = 32
resolution = 16
batch_size = 8
NUM_EXAMPLES = 15#15
TN_HIDDEN_DIM = 512
TN_DROPOUT = 0.1
TN_NHEADS = 8
TN_DIM_FEEDFORWARD = 512
TN_ENC_LAYERS = 3
TN_DEC_LAYERS = 3
with open('files_kr/korean.txt', encoding='utf-8') as f:
    base_alphabet = ''.join(line.strip() for line in f)
ALPHABET = ' ' + base_alphabet
VOCAB_SIZE = len(ALPHABET) + 1
G_LR = 0.00005
D_LR = 0.00005
W_LR = 0.00005
OCR_LR = 0.00005
EPOCHS = 10
NUM_CRITIC_GOCR_TRAIN = 2
NUM_CRITIC_DOCR_TRAIN = 1
NUM_CRITIC_GWL_TRAIN = 2
NUM_CRITIC_DWL_TRAIN = 1
NUM_FID_FREQ = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_SEQ = True
NUM_WORDS = 3
if not IS_SEQ: NUM_WORDS = NUM_EXAMPLES
IS_CYCLE = False
IS_KLD = False
ADD_NOISE = False
ALL_CHARS = False
SAVE_MODEL = 10
SAVE_MODEL_HISTORY = 10

def init_project():
    import os, shutil
    if not os.path.isdir('saved_images'): os.mkdir('saved_images')
    if os.path.isdir(os.path.join('saved_images', EXP_NAME)): shutil.rmtree(os.path.join('saved_images', EXP_NAME))
    os.mkdir(os.path.join('saved_images', EXP_NAME))
    os.mkdir(os.path.join('saved_images', EXP_NAME, 'Real'))
    os.mkdir(os.path.join('saved_images', EXP_NAME, 'Fake'))

