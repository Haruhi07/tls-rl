def concatenate(timeline):
    if len(timeline) != 0:
        cct_timeline = timeline[0]
    else:
        return ''
    for summary in timeline:
        cct_timeline = ' '.join(cct_timeline, summary['text'])
    return cct_timeline

