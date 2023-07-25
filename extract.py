from argparse import ArgumentParser, ArgumentTypeError
import json
from tqdm import tqdm
from subprocess import check_output
from torch import device as torch_device, no_grad as torch_no_grad
from transformers import AutoTokenizer, BertModel, GPT2Model
# import en_core_web_md
from process import parse_sentence
from mapper import Map, deduplication
from spacy import require_gpu as spacy_require_gpu, load as spacy_load


num_processes = 12


def get_line_count(filename):
    return int(check_output(['wc', '-l', filename]).split()[0])


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


@torch_no_grad()
def main():
    parser = ArgumentParser(description='Process lines of text corpus into knowledgraph')
    parser.add_argument('input_filename', type=str, help='text file as input')
    parser.add_argument('output_filename', type=str, help='output text file')
    parser.add_argument('--language_model',default='bert-base-cased',
                        choices=[ 'bert-large-uncased', 'bert-large-cased', 'bert-base-uncased', 'bert-base-cased', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='which language model to use')
    parser.add_argument('--device', default='cuda',
                            type=str, nargs='?',
                            help="Device String")
    parser.add_argument('--include_text_output', default=False,
                            type=str2bool, nargs='?',
                            help="Include original sentence in output")
    parser.add_argument('--threshold', default=0.003,
                            type=float, help="Any attention score lower than this is removed")

    args = parser.parse_args()

    device = torch_device(args.device)
    print('Using device:', device)
    # nlp = en_core_web_md.load()

    '''Create
    Tested language model:

    1. bert-base-cased

    2. gpt2-medium

    Basically any model that belongs to this family should work

    '''

    language_model = args.language_model

    gpu = spacy_require_gpu()
    print(f'SpaCy GPU: {gpu}')
    nlp = spacy_load('en_core_web_md', disable=['tagger', 'ner', 'lemmatizer', 'textcat'])
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    if 'gpt2' in language_model:
        encoder = GPT2Model.from_pretrained(language_model)
    else:
        encoder = BertModel.from_pretrained(language_model)
    encoder.eval()
    encoder = encoder.to(device)
    input_filename = args.input_filename
    output_filename = args.output_filename
    include_sentence = args.include_text_output

    line_count = get_line_count(input_filename)

    with open(input_filename, 'r') as f, open(output_filename, 'w') as g:
        for idx, line in enumerate(tqdm(f, total=line_count)):
            # print(f'Processing line {idx}: {line}')
            sentence = line.strip()
            if len(sentence):
                valid_triplets = []
                # print(f'nlp')
                for sent in nlp(sentence).sents:
                    # Match
                    # print(f'match: parse_sentence: START for sentence {sent.text}')
                    for triplets in parse_sentence(sent.text, tokenizer, encoder, nlp, device, num_processes=num_processes):
                        valid_triplets.append(triplets)
                    # print(f'match: parse_sentence: END')
                if len(valid_triplets) > 0:
                    # Map
                    mapped_triplets = []
                    for triplet in valid_triplets:
                        head = triplet['h']
                        tail = triplet['t']
                        relations = triplet['r']
                        conf = triplet['c']
                        if conf < args.threshold:
                            continue
                        # print(f'map: Map: START')
                        mapped_triplet = Map(head, relations, tail)
                        # print(f'map: Map: END')
                        if 'h' in mapped_triplet:
                            mapped_triplet['c'] = conf
                            mapped_triplets.append(mapped_triplet)
                    # print(f'map: deduplication: START')
                    output = { 'line': idx, 'tri': deduplication(mapped_triplets) }
                    # print(f'map: deduplication: END')

                    if include_sentence:
                        output['sent'] = sentence
                    if len(output['tri']) > 0:
                        g.write(json.dumps( output )+'\n')

if __name__ == '__main__':
    main()