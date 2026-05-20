def word_search(doc_list, keyword):
    ret = []
    for idx in range(len(doc_list)):
        for s in doc_list[idx].split(" "):
            filtered = s.strip(',').strip('.').lower()
            if filtered == keyword.lower():
                ret.append(idx)
                break
    return ret