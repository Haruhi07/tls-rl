def concatenate(timeline):
    cct_timeline = ''

    for summary in timeline:
        cct_timeline = cct_timeline + ' ' + summary['text'][0]

    return cct_timeline

