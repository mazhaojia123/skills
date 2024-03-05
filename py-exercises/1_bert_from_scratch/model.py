import pandas as pd

class IMDBBertDataset(Dataset):
    # Define Special tokens as attributes of class
    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    UNK = '[UNK]'

    MASK_PERCENTAGE = 0.15  # How much words to mask

    MASKED_INDICES_COLUMN = 'masked_indices'
    TARGET_COLUMN = 'indices'
    NSP_TARGET_COLUMN = 'is_next'
    TOKEN_MASK_COLUMN = 'token_mask'

    OPTIMAL_LENGTH_PERCENTILE = 70

    def __init__(self, path, ds_from=None, ds_to=None, should_include_text=False):
        self.ds: pd.Series = pd.read_csv(path)['review']

        if ds_from is not None or ds_to is not None:
            self.ds = self.ds[ds_from:ds_to]

        self.tokenizer = get_tokenizer('basic_english')
        self.counter = Counter()
        self.vocab = None

        self.optimal_sentence_length = None
        self.should_include_text = should_include_text

        if should_include_text:
            self.columns = ['masked_sentence', self.MASKED_INDICES_COLUMN, 'sentence', self.TARGET_COLUMN,
                            self.TOKEN_MASK_COLUMN,
                            self.NSP_TARGET_COLUMN]
        else:
            self.columns = [self.MASKED_INDICES_COLUMN, self.TARGET_COLUMN, self.TOKEN_MASK_COLUMN,
                            self.NSP_TARGET_COLUMN]
        self.df = self.prepare_dataset()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        ...
    
    def prepare_dataset() -> pd.DataFrame:
        ...