import numpy as np

class SumTree(object):
    """
    cấu trúc dữ liệu cây tổng nhị phân (binary sum tree) 
    dùng cho kỹ thuật prioritized experience replay (per).
    """
    def __init__(self, capacity):
        # sức chứa (capacity) phải là một số chẵn và lớn hơn 0
        self.capacity = capacity
        # cây nhị phân có tổng cộng (2 * capacity - 1) node
        self.tree = np.zeros(2 * capacity - 1)
        # mảng lưu trữ dữ liệu thực tế
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0

    def add(self, p, data):
        """thêm dữ liệu mới cùng với độ ưu tiên (priority) p của nó"""
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0 # ghi đè nếu vượt quá sức chứa

    def update(self, tree_idx, p):
        """cập nhật độ ưu tiên của một node lá và lan truyền lên gốc"""
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        
        # lan truyền sự thay đổi (change) lên các node cha
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """lấy ra một node lá dựa trên giá trị ngẫu nhiên v"""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # nếu đã đi xuống tới node lá
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # tìm kiếm xuống dưới: trái hay phải?
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_p(self):
        """trả về tổng độ ưu tiên của toàn bộ cây (nằm ở root node)"""
        return self.tree[0]
