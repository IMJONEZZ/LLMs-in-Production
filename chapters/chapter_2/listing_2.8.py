import numpy as np
from scipy.special import softmax, logsumexp

print("Step 1: Input : 3 inputs, d_model=4")
x = np.array(
    [[1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 2.0], [1.0, 1.0, 1.0, 1.0]]
)
print("x:", x)

print("Step 2: weights 3 dimensions x d_model=4")
w_query = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 1]])
print("w_query:", w_query)

w_key = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 1, 0]])
print("w_key:", w_key)

w_value = np.array([[0, 2, 0], [0, 3, 0], [1, 0, 3], [1, 1, 0]])
print("w_value:", w_value)

print("Step 3: Matrix multiplication to obtain Q,K,V")
print("Query: x * w_query")
Q = np.matmul(x, w_query)
print("Q:", Q)

print("Key: x * w_key")
K = np.matmul(x, w_key)
print("K:", K)

print("Value: x * w_value")
V = np.matmul(x, w_value)
print("V:", V)

print("Step 4: Scaled Attention Scores")
k_d = 1  # This is normally the square root of the number of dimensions
attention_scores = (Q @ K.transpose()) / k_d
print(attention_scores)

print("Step 5: Scaled softmax attention_scores for each vector")
attention_scores[0] = softmax(attention_scores[0])
attention_scores[1] = softmax(attention_scores[1])
attention_scores[2] = softmax(attention_scores[2])
print(attention_scores[0])
print(attention_scores[1])
print(attention_scores[2])

print("Step 6: attention value obtained by score1/k_d * V")
print(V[0])
print(V[1])
print(V[2])
attention1 = attention_scores[0].reshape(-1, 1)
attention1 = attention_scores[0][0] * V[0]
print("Attention 1:", attention1)

attention2 = attention_scores[0][1] * V[1]
print("Attention 2:", attention2)

attention3 = attention_scores[0][2] * V[2]
print("Attention 3:", attention3)

print(
    "Step 7: sum the results to create the first line of the output matrix"
)
attention_input1 = attention1 + attention2 + attention3
print(attention_input1)

print("Step 8: Step 1 to 7 for inputs 1 to 3")
# This assumes that we actually went through the whole process for all 3
# We'll just take a random matrix of the correct dimensions in lieu
attention_head1 = np.random.random((3, 64))
print(attention_head1)

print("Step 9: We assume we trained the 8 heads of the attention sub-layer")
z0h1 = np.random.random((3, 64))
z1h2 = np.random.random((3, 64))
z2h3 = np.random.random((3, 64))
z3h4 = np.random.random((3, 64))
z4h5 = np.random.random((3, 64))
z5h6 = np.random.random((3, 64))
z6h7 = np.random.random((3, 64))
z7h8 = np.random.random((3, 64))
print("shape of one head", z0h1.shape, "dimension of 8 heads", 64 * 8)

print(
    "Step 10: Concatenate heads 1 to 8 to get the original 8x64=512 output dim"
)
output_attention = np.hstack(
    (z0h1, z1h2, z2h3, z3h4, z4h5, z5h6, z6h7, z7h8)
)
print(output_attention)


def DotProductAttention(query, key, value, mask, scale=True):
    """Dot product self-attention.
    Args:
        query: array of query representations with shape (L_q by d)
        key: array of key representations with shape (L_k by d)
        value: array of value representations with shape (L_k by d) where L_v = L_k
        mask: attention-mask, gates attention with shape (L_q by L_k)
        scale: whether to scale the dot product of the query and transposed key

    Returns:
        numpy.ndarray: Self-attention array for q, k, v arrays. (L_q by L_k)
    """

    assert (
        query.shape[-1] == key.shape[-1] == value.shape[-1]
    ), "Embedding dimensions of q, k, v aren't all the same"

    # Save dimension of the query embedding for scaling down the dot product
    if scale:
        depth = query.shape[-1]
    else:
        depth = 1

    # Calculate scaled query key dot product according to formula above
    dots = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(depth)

    # Apply the mask
    if mask is not None:
        dots = np.where(mask, dots, np.full_like(dots, -1e9))

    # Softmax formula implementation
    logsumexpo = logsumexp(dots, axis=-1, keepdims=True)

    # Take exponential of dots minus logsumexp to get softmax
    dots = np.exp(dots - logsumexpo)

    # Multiply dots by value to get self-attention
    attention = np.matmul(dots, value)

    return attention


def masked_dot_product_self_attention(q, k, v, scale=True):
    """Masked dot product self attention.
    Args:
        q: queries.
        k: keys.
        v: values.
    Returns:
        numpy.ndarray: masked dot product self attention tensor.
    """

    # Size of the penultimate dimension of the query
    mask_size = q.shape[-2]

    # Creates ones below the diagonal and 0s above shape (1, mask_size, mask_size)
    mask = np.tril(np.ones((1, mask_size, mask_size), dtype=np.bool_), k=0)

    return DotProductAttention(q, k, v, mask, scale=scale)
