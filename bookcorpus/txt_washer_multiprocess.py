import os
from multiprocessing import Pool, cpu_count

txt_path = "./romance_txt"
epub_path = './Romance_epub'
results_path = './washed_romance/'

# TODO: Add your own rule here
def replace(sentence):
    return sentence.replace(". . .", " ... ").replace("``", "").replace(" ' ", "").replace(" ’ ", "").replace(" '", \
             "").replace("' ","").replace("'", "").replace("”", "").replace("“", "").replace(" `` ", "").replace("''", "").replace(" nt ", "nt ")

# skiphead and skiptail are hyperparameters used to screen out the content-irrelavant text
def handle_epubtxt(txt_filename, skiphead=100, skiptail=200):

    with open(os.path.join(txt_path, txt_filename)) as f:
        content = f.readlines()
        i = 0

        # Uncomment to skip forehead part before first "Chapter" (not guaranteed to be Chapter 1 but works sometime)
        # while i < len(content) and "chapter" not in content[i].lower():
        #     i += 1

        # TODO: Add your own rule here
        while i < len(content):
            if "* Chapter" in content[i]:
                content = content[:i] + content[i + 3:]
            elif ".png" in content[i] or ".jpg" in content[i]:
                print(f"{txt_filename} abandoned!")
                return
            elif "*" in content[i]:
                content[i] = content[i].replace("*", "")
            elif "♥" in content[i]:
                content[i] = content[i].replace("♥", "")
                # content = content[:i] + content[i + 1:]
            elif "#" in content[i]:
                content[i] = content[i].replace("#", "")
            else:
                i += 1

        new_contents = []
        passages = []
        for cur_idx, line in enumerate(content):
            if not line.split():
                new_passage = " ".join(passages).strip()
                if new_passage.split():
                    new_contents.append(new_passage)
                passages = []
            else:
                passages.append(line.strip())
        new_contents = new_contents[skiphead:-skiptail]
        new_contents.append(" ".join(passages).strip())
        new_content = replace("\n".join(new_contents))

        new_file = os.path.join(results_path, f"{txt_filename}")
        with open(new_file, 'w') as newfile:
            newfile.write(new_content)
            newfile.close()

def handle_txt(txt_filename, skiphead=100, skiptail=100):
    with open(os.path.join(txt_path, txt_filename)) as f:
        content = f.readlines()
        i = 0
        while i < len(content) and "chapter" not in content[i].lower():
            i += 1

        new_contents = []
        for cur_idx, line in enumerate(content):
            if line.split():
                new_contents.append(line.strip())
        new_contents = new_contents[skiphead:-skiptail]

        new_file = os.path.join(results_path, f"{txt_filename}")
        with open(new_file, 'w') as newfile:
            newfile.write("\n".join(new_contents))
            newfile.close()

if not os.path.exists(results_path):
    os.mkdir(results_path)

for root, dirs, files in os.walk(epub_path, topdown=False):
    epub_files = [fname.split('.')[0] for fname in files]

txts, txt_from_epub = [], []
for root, dirs, files in os.walk(txt_path, topdown=False):
    for txt_filename in files:
        filename = txt_filename.split('__')[1].split('.')[0]
        if filename in epub_files:
            txt_from_epub.append(txt_filename)
        else:
            txts.append(txt_filename)

cpu_count = cpu_count()
pool = Pool(cpu_count)
pool.map(handle_txt, txts)
pool.map(handle_epubtxt, txt_from_epub)
