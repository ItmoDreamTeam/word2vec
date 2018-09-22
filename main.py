import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import itertools
import time


def log_progress(sequence, name, every=10):
    for index, item in enumerate(sequence):
        if index % every == 0:
            print(name, item)
        yield item


# Preparation
def decode(txt):
    return words_sorted_by_frequency[txt] if isinstance(txt, (int, np.integer)) else list(map(decode, list(txt)))


def random_context(text, distance_between_words=5):
    random_sentence = text[np.random.randint(0, len(text))]
    word_index = np.random.randint(0, len(random_sentence))
    word = random_sentence[word_index]
    left_context = random_sentence[max(0, word_index - distance_between_words):word_index]
    right_context = random_sentence[(word_index + 1):min(len(random_sentence), word_index + distance_between_words)]
    return word, left_context + right_context


source_str = open('AK.txt', 'r').read()
source_words = re.findall('(\w+)|\n\n+|[.?!]', source_str, re.UNICODE)
print('Список слов:', source_words[:10])
print('Всего слов, включая концы предложений:', len(source_words))

all_words = Counter(source_words)
number_of_sentences = all_words.pop('', None)
print('Всего предложений:', number_of_sentences)
print('Всего различных слов:', len(all_words))
print('Самые часто встречающиеся слова:', all_words.most_common(10))

sentences = [list(y) for x, y in itertools.groupby(source_words, lambda z: z == '') if not x]
for sentence in sentences:
    first_word_lower_case = sentence[0].lower()
    if first_word_lower_case in all_words: sentence[0] = first_word_lower_case
print('Начало текста:', sentences[:6])

all_words = Counter([word for sentence in sentences for word in sentence])
number_of_sentences = len(sentences)
print('Всего предложений:', number_of_sentences)
print('Всего различных слов:', len(all_words))
print('Самые часто встречающиеся слова:', all_words.most_common(10))

least_count = 50
max_count = 10000
common_words = {k: v for k, v in all_words.items() if v >= least_count and v <= max_count}
print('Число встречающихся минимум', least_count, "раз слов:", len(common_words))

words_sorted_by_frequency = list(map(lambda x: x[0], sorted(common_words.items(), key=lambda x: -x[1])))
words_codes = dict([(w, c) for c, w in enumerate(words_sorted_by_frequency)])
# эти коллекции можно использовать для сопоставления словам чисел и обратно
print(words_sorted_by_frequency[words_codes['ты']])

encoded_text = [[words_codes[word] for word in sentence if word in words_codes] for sentence in sentences]
# убираем пустые предложения меньше заданной длины
minimal_length = 4
encoded_text = [sentence for sentence in encoded_text if len(sentence) >= minimal_length]
print('Начало закодированного текста:', encoded_text[:6])
print('Число предложений:', len(encoded_text))

print('Раскодированное начало:', decode(encoded_text[:6]))
print('Случайный контекст:', decode(random_context(encoded_text)))


# Функции потерь возвращают пару: значение функции, градиент
def loss_function_sq(Dx, y):
    delta = Dx - y
    return np.dot(delta, delta), delta


def loss_function_cr(tx, y):
    return -np.sum(tx * np.log(y)), -tx / y


# Линейные функции на одном слое
def linear_layer(theta, x):
    return np.dot(theta, x)


def linear_layer_dx(dx, theta):
    return np.dot(dx, theta)


def linear_layer_dtheta(dx, x):
    return dx[:, None] * x[None, :]


# Функции активации возращают пару: значение функции на аргументе, вспомогательные значения.
# Вспомогательные значения используется позже для ускорения вычисления градинта.
def logistic_function(x):
    t = np.exp(-x)
    return 1 / (1 + t), t


# Функция для вычисления градиента принимают на вход результат (y,t) вычисления функции активации.
def logistic_function_dx(dx, y, t):
    return dx * t * y * y


# Функция, используемая на выходе нейронной сети, если требуется получить распределение вероятности
def softmax(x):
    t = np.exp(-x)
    return t / np.sum(t)


def softmax_dx(dx, y):
    # y=softmax(x)
    w = y * dx
    return y * np.sum(w) - w


# Neural network
dimensionality = len(words_sorted_by_frequency)


def context_to_vector(context):
    result = np.zeros(dimensionality)
    for word in context: result[word] += 1. / len(context)
    return result


print("Число слов в случайном контексте:", np.sum(0 != context_to_vector(random_context(encoded_text)[1])))


def neural_network(A, B, x):
    return softmax(linear_layer(B, linear_layer(A, x)))


def loss(A, B, x, tx, return_grad=True):
    z3 = linear_layer(A, x)
    z2 = linear_layer(B, z3)
    z1 = softmax(z2)
    R, dz1 = loss_function_cr(tx, z1)
    if not return_grad: return R
    dz2 = softmax_dx(dz1, z1)
    dB = linear_layer_dtheta(dz2, z3)
    dz3 = linear_layer_dx(dz2, B)
    dA = linear_layer_dtheta(dz3, x)
    return R, dA, dB


def grad(A, B, x, tx, return_grad=True):
    z3 = linear_layer(A, x)
    z2 = linear_layer(B, z3)
    z1 = softmax(z2)
    R, dz1 = loss_function_cr(tx, z1)
    if not return_grad: return R
    dz2 = softmax_dx(dz1, z1)
    dz3 = linear_layer_dx(dz2, B)
    dx = linear_layer_dx(dz3, A)
    return R, dx


# Проверка корректности градиента
dim = 10
x = np.random.rand(dim)
t = np.random.rand(dim)
A = np.random.rand(dim, dim)
B = np.random.rand(dim, dim)
DA = np.random.rand(dim, dim)
DB = np.random.rand(dim, dim) * 0
R, dA, dB = loss(A, B, x, t)
derror = np.sum(DA * dA) + np.sum(DB * dB)
epsilon = np.logspace(-8, -1, 10)
error = np.empty(len(epsilon))
for n in range(len(epsilon)):
    error[n] = loss(A + epsilon[n] * DA, B + epsilon[n] * DB, x, t, return_grad=False) - R - derror * epsilon[n]
plt.loglog(epsilon, np.abs(error), '-k')
plt.loglog(epsilon, epsilon * epsilon, '--r')
plt.xlabel("Argument increment")
plt.ylabel("Error")
plt.legend(['Experiment', 'Prediction'], loc=4)
plt.show()


# Training
def make_batch(text, size=300):
    C = []
    W = []
    for _ in range(size):
        word, context = random_context(text)
        W.append(context_to_vector([word]))
        C.append(context_to_vector(context))
    return W, C


def train_on_batch(A, B, batch, number_of_steps=3, step_size=5):
    batch_size = len(batch[0])
    history = []
    for _ in range(number_of_steps):
        dA = np.zeros(A.shape)
        dB = np.zeros(B.shape)
        error = 0
        for W, C in zip(*batch):
            R, DA, DB = loss(A, B, W, C)
            error += R
            dA += DA
            dB += DB
        A -= step_size / batch_size * dA
        B -= step_size / batch_size * dB
        history.append(error / batch_size)
    return history


def train_network(A, B, text, test=None, number_of_steps=1000, debug=False):
    report_each = number_of_steps / 10
    history = []
    if not test is None:
        print("Initial error {}".format(test_network(A, B, test_text)))
    try:
        for n in log_progress(range(1, number_of_steps + 1), name='Batch'):
            error = train_on_batch(A, B, make_batch(text))
            if debug: print(n, ":", error[-1])
            if not test is None and n % report_each == 0:
                print("Epoch {}, generalization error {}".format(n, test_network(A, B, test_text)))
            history.extend(error)
    except KeyboardInterrupt:
        pass
    return history


def test_network(A, B, text, number_of_samples=1000):
    error = 0
    for _ in range(number_of_samples):
        word, context = random_context(text)
        W = context_to_vector([word])
        C = context_to_vector(context)
        error += loss(A, B, W, C, return_grad=False)
    return error / number_of_samples


# Training
ratio = 0.9
train_text = encoded_text[:int(ratio * len(encoded_text))]
test_text = encoded_text[int(ratio * len(encoded_text)):]
features = 50
A = np.random.rand(features, dimensionality)
B = np.random.rand(dimensionality, features)

training_start_time = time.time() * 1000
history = train_network(A, B, train_text, test=test_text)
print('Training lasted for %d ms' % (time.time() * 1000 - training_start_time))

plt.plot(history, ',')
plt.xlabel("Training step")
plt.ylabel("Error")
plt.show()


def word2vec(B, n):
    return B[n, :]


def similar_words(B, vec):
    dist = np.sum((B - vec[None, :]) ** 2, axis=1)
    return sorted(enumerate(dist), key=lambda x: x[1])


def distance_matrix(B):
    return np.sum((B[:, :, None] - B[:, None, :]) ** 2, axis=0)


print(words_sorted_by_frequency)


def show_similar(B, vec, count=10):
    for code, freq in similar_words(B, vec)[:count]:
        print("{}/{}".format(decode(code), int(freq)), end=" ")


# Analysis
words_to_compare = ['Анна', 'Степан', 'давно', 'много', 'руки']
for word in words_to_compare:
    print(word, ":", end=" ")
    show_similar(B, word2vec(B, words_codes[word]))
    print()


# Попробуем проанализировать отношения слов
def show_relative(n1, n2, n3):
    w1 = words_codes[n1]
    v1 = word2vec(B, w1)
    w2 = words_codes[n2]
    v2 = word2vec(B, w2)
    w3 = words_codes[n3]
    v3 = word2vec(B, w3)
    v4 = v2 - v1 + v3
    print("Как '{}' относится к '{}', также к '{}' относятся следующие слова в порядке убывания уверенности:".format(
        decode(w1), decode(w2), decode(w3)))
    show_similar(B, v4)
    print("\n")


show_relative('Анна', 'Степан', 'она')
show_relative('Степан', 'Анна', 'он')

# Выведем все вектора слов
plt.imshow(np.abs(B.T), interpolation='none')
plt.show()

# Выведем матрицу расстояний между словами
plt.imshow(distance_matrix(B), interpolation='none')
plt.show()

n1 = 'он'
n2 = 'она'
ns = ['Анна', 'Степан', 'жена', 'хорошо', 'руки']
w1 = words_codes[n1]
w2 = words_codes[n2]
v1 = word2vec(B, w1)
v2 = word2vec(B, w2)
p1 = np.dot(B, v1)
p2 = np.dot(B, v2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(p1, p2, '.')
for n in ns:
    w = words_codes[n]
    ax.plot(p1[w], p2[w], '.r')
    ax.annotate(n, xy=(p1[w], p2[w]), color='r')
plt.xlabel(n1)
plt.ylabel(n2)
plt.show()
