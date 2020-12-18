import pandas as pd


def sanity_check(tr, va, format='dataframe'):
    if format == 'dataframe':
        # remove nan...
        tr_text = [x for x in tr['text'] if isinstance(x, str)]
        va_text = [x for x in va['text'] if isinstance(x, str)]
    elif format == 'list':
        tr_text = tr
        va_text = va
    else:
        raise NotImplementedError

    # first, check all are lowercase
    #for utterance in tr_text:
    #    assert not any(x.isupper() and ord(x) < 128 for x in utterance), \
    #        f'tr set contains capitalized letters: {utterance}'
    #for utterance in va_text:
    #    assert not any(x.isupper() and ord(x) < 128 for x in utterance), \
    #        f'va set contains capitalized letters: {utterance}'

    # then, check overlap (soft warning)
    overlap = set(tr_text) & set(va_text)
    if overlap:
        print('*'*50 + 'WARNING' + '*'*50)
        print(
            f'va and tr has {len(overlap)} overlap: {set(tr_text) & set(va_text)}')


if __name__ == '__main__':
    df_tr = pd.read_csv(
        'data/2k_top_n_extended_merged_train.csv', lineterminator='\n')
    df_va = pd.read_csv(
        'data/2k_top_n_extended_merged_eval.csv', lineterminator='\n')
    sanity_check(df_tr, df_va)
