import datetime
import os
import json
import pathlib
import subprocess
from argparse import ArgumentParser
from xml.etree import ElementTree


def annotate(source_article_path, heideltime_path):
    apply_heideltime = heideltime_path / 'apply-heideltime.jar'
    heideltime_config = heideltime_path / 'config.props'
    subprocess.run([
        'java',
        '-jar',
        str(apply_heideltime),
        str(heideltime_config),
        str(source_article_path),
        'txt'
    ])


def annotate_articles(source_path, heideltime_path):
    for topic in sorted(os.listdir(source_path)):
        print("Annotating articles in topic: {}".format(topic))

        date_path = source_path / topic / "InputDocs"
        for date in sorted(os.listdir(date_path)):
            articles_path = date_path / date
            if not articles_path.is_dir():
                continue
            annotated = False
            for file in os.listdir(articles_path):
                if 'timeml' in file:
                    annotated = True
                    break
            if not annotated:
                annotate(articles_path, heideltime_path)


def extract_dates(annotated_text):
    # cleanup heideltime bugs
    replace_pairs = [
        ("T24", "T12"),
        (")TMO", "TMO"),
        (")TAF", "TAF"),
        (")TEV", "TEV"),
        (")TNI", "TNI"),
    ]
    for old, new in replace_pairs:
        annotated_text = annotated_text.replace(old, new)

    time_values = []

    try:
        root = ElementTree.fromstring(annotated_text)
    except ElementTree.ParseError as e:
        return None, None

    def extract_time_tag_value(time_tag):
        ret = None

        if 'type' not in time_tag.attrib:
            return ret
        elif time_tag.attrib['type'] == 'DATE':
            formats = ['%Y-%m-%d', '%Y-%m', '%Y']
        elif time_tag.attrib['type'] == 'TIME':
            formats = ['%Y-%m-%dT%H:%M', '%Y-%m-%dTMO', '%Y-%m-%dTEV', '%Y-%m-%dTNI', '%Y-%m-%dTAF']
        else:
            return ret

        for time_format in formats:
            try:
                ret = datetime.datetime.strptime(time_tag.attrib['value'], time_format)
            except:
                pass
        return ret

    for time_tag in root:
        if time_tag.text is None:
            continue
        value = extract_time_tag_value(time_tag)
        if value != None:
            v = str(value).split(' ')[0]
            time_values.append(v)

    return time_values


def convert_articles(source_path, target_path):
    for topic in sorted(os.listdir(source_path)):
        print("Converting articles in topic: {}".format(topic))
        article_list = []

        target_topic_path = target_path / topic
        if not target_topic_path.exists():
            os.mkdir(target_topic_path)

        target_json_path = target_topic_path / "articles.json"
        target_json = open(target_json_path, "w")
        date_path = source_path / topic / "InputDocs"

        for date in sorted(os.listdir(date_path)):
            articles_path = date_path / date
            if not articles_path.is_dir():
                continue

            for article in sorted(os.listdir(articles_path)):
                if "timeml" in article:
                    continue
                source_article_path = articles_path / article
                source_annotated_article_path = pathlib.Path(str(source_article_path) + ".timeml")
                source_article = open(source_article_path, "r")
                source_annotated_article = open(source_annotated_article_path, "r")
                uid = article[:-8]
                text = source_article.read().replace('\n', '')
                annotated_text = source_annotated_article.read()
                dct = date
                dates = extract_dates(annotated_text)

                data = {"uid": uid, "dct": dct, "dates": dates, "text": text}
                article_list.append(data)

        json.dump(article_list, target_json)
        target_json.close()


def convert_timelines(source_path, target_path):
    for topic in sorted(os.listdir(source_path)):
        print("Converting timelines in topic: {}".format(topic))

        target_topic_path = target_path / topic
        if not target_topic_path.exists():
            os.mkdir(target_topic_path)

        source_timelines_path = source_path / topic / "timelines"
        for timeline in sorted(os.listdir(source_timelines_path)):
            if timeline == ".DS_Store":
                continue
            source_timeline_path = source_timelines_path / timeline
            target_timeline_json_name = timeline + ".timeline.json"
            target_timeline_json_path = target_topic_path / target_timeline_json_name
            if target_timeline_json_path.exists():
                continue
            with open(source_timeline_path, "r") as source_timeline:
                timeline_text = source_timeline.readlines()
            timeline_list = []
            date = ''
            text = ''
            for line in timeline_text:
                tmp = line.split('-')
                if len(tmp) == 3 and len(tmp[0]) == 4 and len(tmp[1]) == 2 and len(tmp[2]) == 3:
                    date = line.replace('\n', '')
                elif len(tmp) == 33:
                    text = text.replace('\n', '')
                    date_with_summary = {'date': date, 'text': text}
                    text = ""
                    timeline_list.append(date_with_summary)
                else:
                    text += ' ' + line.replace('\n', '')
            with open(target_timeline_json_path, "w") as target_timeline_json:
                json.dump(timeline_list, target_timeline_json)


def main():
    parser = ArgumentParser()
    # Load arguments
    parser.add_argument("--heideltime", required=True, help="location of heideltime")
    parser.add_argument("--source", type=str, default="Dataset/tls-rl-dataset/t17")
    parser.add_argument("--target", type=str, default="./dataset/t17")
    args = parser.parse_args()

    # source_path is an absolute path starting from ~/
    source_path = pathlib.Path.home() / args.source
    target_path = pathlib.Path(args.target)
    heideltime_path = pathlib.Path(args.heideltime)

    if not source_path.exists():
        raise FileNotFoundError('Dataset not found in {}'.format(args.source))
    if not target_path.exists():
        target_path.mkdir(exist_ok=True)

    annotate_articles(source_path, heideltime_path)
    convert_articles(source_path, target_path)
    convert_timelines(source_path, target_path)


if __name__ == "__main__":
    main()
