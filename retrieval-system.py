import glob
import re
import numpy
import operator


def main():
    query_path = "/home/kecs/Documents/Coding/RetrievalSystem/Query.txt"
    documents = get_documents("/home/kecs/Documents/Coding/RetrievalSystem/")

    # build atlas containing all words of all documents
    atlas = []
    for document in documents:
        path = documents[document]
        document_atlas = tokenize(path)
        for token in document_atlas:
            atlas.append(token)

    # build tfidf of query half
    query_matrix = build_matrix(query_path, atlas)

    similarities = []
    # build tfidf of documents and calculate the similarities of query and each document respectively
    for document in documents:
        path = documents[document]
        document_matrix = build_matrix(path, atlas)
        similarities.append(calculate_similarity(query_matrix, document_matrix))

    # sort similarities and retrieve the most relevant

    files = list(documents.keys())

    results = {}
    for i in range(len(files)):
        results[files[i]] = similarities[i]

    sorted_results = sorted(results.items(), key=operator.itemgetter(1))
    sorted_results.reverse()

    print("The most relevant documents to the query in descending order of relevance:")
    for i in range(len(sorted_results)):
        print(sorted_results[i][0])


def get_documents(path_to_document_folder):
    # get all files with the ".txt" ending
    regex_for_txt = path_to_document_folder + "*.txt"
    file_paths = glob.glob(regex_for_txt)

    # get the names of the individual files
    file_names = []
    for file_name in file_paths:
        file_names.append(re.sub("^" + path_to_document_folder, "", file_name))

    # put the files and their respective paths in a dictionary
    documents = {}
    for file_name in file_names:
        documents[file_name] = file_paths[file_names.index(file_name)]

    # remove the Query from the documents
    del documents["Query.txt"]

    return documents


def build_matrix(path, atlas):
    matrix = {}
    for token in atlas:
        matrix[token] = []

        term_frequency = calculate_term_frequency(token, path)
        matrix[token].append(term_frequency)

        normalized_term_frequency = calculate_normalized_term_frequency(term_frequency)
        matrix[token].append(normalized_term_frequency)

        document_frequency = calculate_document_frequency()
        matrix[token].append(document_frequency)

        inverse_document_frequency = calculate_inverse_document_frequency(1, document_frequency)
        matrix[token].append(inverse_document_frequency)

        tfidf = calculate_tfidf(normalized_term_frequency, inverse_document_frequency)
        matrix[token].append(tfidf)
    return matrix


def calculate_term_frequency(token, path):
    file = open(path, 'r')
    corpus = file.read()

    term_frequency = 0

    # scan document for occurrence
    length_of_corpus = len(corpus)
    length_of_token = len(token)

    for i in range(0, (length_of_corpus - length_of_token) + 1):
        buffer = corpus[i: i + length_of_token]
        if buffer == token:
            term_frequency += 1
    return term_frequency


def calculate_normalized_term_frequency(term_frequency):
    if term_frequency == 0:
        return 0
    return numpy.log10(term_frequency) + 1


def calculate_document_frequency():
    return 1


def calculate_inverse_document_frequency(number_of_documents, document_frequency):
    return numpy.log10(number_of_documents / document_frequency) + 1


def calculate_tfidf(normalized_term_frequency, inverse_document_frequency):
    return normalized_term_frequency * inverse_document_frequency


def tokenize(file_path):
    file = open(file_path, 'r')
    input_string = file.read()
    tokens = input_string.split(' ')
    return tokens


def calculate_similarity(query_matrix, document_matrix):
    query_vector = build_tfidf_vector(query_matrix)
    document_vector = build_tfidf_vector(document_matrix)
    similarity = numpy.dot(query_vector, document_vector)
    return similarity


def build_tfidf_vector(matrix):
    vector = []
    for entry in matrix.values():
        vector.append(entry[4])
    return vector


if __name__ == "__main__":
    main()
