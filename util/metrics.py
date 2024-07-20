import numpy as np


def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) score between two bounding boxes.
    
    Args:
    box1, box2: List of four integers [x1, y1, x2, y2] representing the bounding box coordinates.
    
    Returns:
    iou: IoU score as a float.
    """
    
    # Unpack the coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Compute the coordinates of the intersection rectangle
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # Compute the area of the intersection rectangle
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Compute the area of both bounding boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Compute the area of the union
    union_area = box1_area + box2_area - inter_area
    
    # Compute the IoU
    iou = inter_area / union_area
    
    return iou

def compute_PNLS(seq1, seq2, match=1, mismatch=-1, gap=-1):

    """
    Computes the Partial Normalized Levenshtein Similarity score between two bounding boxes.
    
    Args:
    seq1 (true string), seq2 (model prediction)
    
    Returns:
    pnls: pnls score as a float.
    """

    if seq1 == '':
        return 1
    
    # Initialize the scoring matrix
    m, n = len(seq1), len(seq2)
    score_matrix = [[0] * (n + 1) for _ in range(m + 1)]
    opt_pos = 0
    opt_v = -1

    # Initialize the gap penalties
    for i in range(1, m + 1):
        score_matrix[i][0] = gap * i
    # for j in range(1, n + 1):
    #     score_matrix[0][j] = gap * j

    # Fill the scoring matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = score_matrix[i - 1][j - 1] + (match if seq1[i - 1] == seq2[j - 1] else mismatch)
            delete_score = score_matrix[i - 1][j] + gap
            insert_score = score_matrix[i][j - 1] + gap
            score_matrix[i][j] = max(match_score, delete_score, insert_score)

            if i == m and score_matrix[m][j] > opt_v:
                opt_v = score_matrix[m][j]
                opt_pos = j

    # Traceback to find the alignment
    align1, align2 = '', ''
    i, j = m, opt_pos
    while i > 0 and j > 0:
        current_score = score_matrix[i][j]
        diagonal_score = score_matrix[i - 1][j - 1]
        up_score = score_matrix[i - 1][j]
        left_score = score_matrix[i][j - 1]

        if current_score == diagonal_score + (match if seq1[i - 1] == seq2[j - 1] else mismatch):
            align1 = seq1[i - 1] + align1
            align2 = seq2[j - 1] + align2
            i -= 1
            j -= 1
        elif current_score == up_score + gap:
            align1 = seq1[i - 1] + align1
            align2 = '-' + align2
            i -= 1
        else:
            align1 = '-' + align1
            align2 = seq2[j - 1] + align2
            j -= 1

    while i > 0:
        align1 = seq1[i - 1] + align1
        align2 = '-' + align2
        i -= 1

    while j > 0:
        align1 = '-' + align1
        align2 = seq2[j - 1] + align2
        j -= 1

    # Remove leading and trailing gaps from the aligned sequences
    start_index = align1.find(seq1[0])
    end_index = align1.rfind(seq1[-1]) + 1
    trimmed_align1 = align1[start_index:end_index]
    trimmed_align2 = align2[start_index:end_index]

    align_string = ''.join(['|' if x == y else ' ' for x, y in zip(trimmed_align1, trimmed_align2)])
    
    # print(trimmed_align1)
    # print(align_string)
    # print(trimmed_align2)

    
    pnls = sum([1 for x in align_string if x == '|'])

    # return trimmed_align1, trimmed_align2, pnls
    return pnls / len(trimmed_align1)
