# import jsonlines
# from collections import defaultdict
# import json
#
# book_dict = defaultdict(list) #
# with jsonlines.open('url_list.jsonl') as reader:
#     for obj in reader:
#         for cate in [g.split(':')[1].split("»")[1].strip() for g in obj['genres'] if "»" in g]:
#             book_dict[cate].append(obj['b_idx'])
#
# # with jsonlines.open('bookdict.json', mode='w') as writer:
# #     writer.write(book_dict)
#
# with open('test.txt', 'w') as t:
#     t.write("\n".join(sorted([f"{num} {cate}" for cate, num in zip(list(book_dict.keys()), list(map(str, map(len, book_dict.values()))))], key=lambda x: int(x.split(' ')[0]))))

# with open(f'{genre}.json', 'w') as t:
#     t.write("\n".join(sorted([f"{num} {cate}" for cate, num in zip(list(book_dict.keys()), list(map(str, map(len, book_dict.values()))))], key=lambda x: int(x.split(' ')[0]))))

import jsonlines
import json

genre = 'Young adult or teen'

with open(f'{genre}.jsonl', 'w') as t:
    with jsonlines.open('url_list.jsonl') as reader:
        for obj in reader:
            for cate in [g.split(':')[1].split("»")[1].strip() for g in obj['genres'] if "»" in g]:
                if genre in cate:
                    json.dump(obj, t)
                    t.write('\n')
                    break