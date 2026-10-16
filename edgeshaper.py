### Module implementing EdgeSHAPer algorithm ###
### Author: Andrea Mastropietro © All rights reserved ###

import torch # Thư việnpytorch
import torch.nn.functional as F # Thư viện pytorch

import numpy as np # Thư viện numpy
from numpy.random import default_rng # Thư viện numpy

import matplotlib.pyplot as plt # Thư viện matplotlib
# from tqdm import tqdm
from tqdm import tqdm # Thư viện tqdm

from rdkit import Chem # Thư viện rdkit
from rdkit.Chem import Draw # Thư viện rdkit
from rdkit_heatmaps import mapvalues2mol # Thư viện rdkit_heatmaps
from rdkit_heatmaps.utils import transform2png # Thư viện rdkit_heatmaps

###EdgeSHAPer as a class###

class Edgeshaper(): # Định nghĩa class Edgeshaper
    """
Lớp EdgeSHAPer dùng để xấp xỉ giá trị Shapley, đo mức độ quan trọng của các cạnh trong GNN.

    P (float, tùy chọn): xác suất để một cạnh tồn tại trong đồ thị ngẫu nhiên.
        Mặc định là mật độ của đồ thị gốc.
    M (int): số bước lấy mẫu Monte Carlo cần thực hiện.
    target_class (int, tùy chọn): chỉ số của lớp dự đoán cần giải thích.
        Mặc định là lớp 0. Đặt None nếu mô hình là mô hình hồi quy.
    deviation (float, tùy chọn): ngưỡng chênh lệch cho phép giữa tổng các giá trị Shapley
        và xác suất đầu ra dự đoán. Nếu là None, sẽ chạy đủ M bước lấy mẫu; nếu không,
        quá trình dừng khi đạt được độ xấp xỉ mong muốn (deviation). Tuy nhiên, sau M bước
        quá trình luôn kết thúc.
    log_odds (bool, tùy chọn): Mặc định là False. Nếu là True, dùng log odds thay cho
        xác suất làm giá trị để xấp xỉ Shapley.
    seed (float, tùy chọn): seed cho bộ sinh số ngẫu nhiên.
    device (string, tùy chọn): cho biết dùng cuda hay cpu.

Trả về:
    list: danh sách các giá trị Shapley của các cạnh do EdgeSHAPer tính ra.
        Thứ tự các cạnh giống như trong E.
"""

    def __init__(self, model, x, edge_index, edge_weight = None, device = "cpu"): # Khởi tạo class Edgeshaper
        """
Tham số:
    model (Torch.NN.Module): mô hình GNN của PyTorch được sử dụng.
    x (tensor): tensor 2 chiều chứa đặc trưng của các nút trong đồ thị cần giải thích.
    edge_index (tensor): edge index của đồ thị cần giải thích.
    edge_weight (tensor): trọng số của các cạnh.
    device (string, tùy chọn): cho biết dùng cuda hay cpu.
"""
        super(Edgeshaper, self).__init__()
        # torch.manual_seed(12345) # Thiết lập seed để đảm bảo kết quả tính toán được lặp lại.
        self.model = model # Mô hình GNN.
        self.model.to(device) # Chuyển mô hình GNN sang device (cuda hoặc cpu). 
        self.x = x.to(device) # Chuyển tensor x sang device (cuda hoặc cpu). 
        self.edge_index = edge_index.to(device) # Chuyển tensor edge_index sang device (cuda hoặc cpu). 
        self.edge_weight = edge_weight.to(device) if edge_weight is not None else None # Chuyển tensor edge_weight sang device (cuda hoặc cpu).
        self.device = device # Thiết lập device (cuda hoặc cpu).


        self.phi_edges = None # Giá trị Shapley của các cạnh.
        
        self.target_class = None # Chỉ số của lớp dự đoán cần giải thích.
        self.explained = False # Đã giải thích hay chưa.

        self.pertinent_positive_set = None # Tập hợp các cạnh quan trọng.
        self.minimal_top_k_set = None # Tập hợp các cạnh quan trọng tối thiểu.
        self.minimal_top_k_set = None # Tập hợp các cạnh quan trọng tối thiểu.
        self.fidelity = None # Độ trung thực.
        self.infidelity = None # Độ không trung thực.
        self.trustuworthiness = None # Độ tin cậy.

        self.original_pred_prob = None # Xác suất dự đoán ban đầu của mô hình GNN.

    def explain(self, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = None): # Hàm giải thích bằng EdgeSHAPer
        """
Tính mức độ quan trọng của các cạnh bằng thuật toán EdgeSHAPer.

    M (int): số bước lấy mẫu Monte Carlo cần thực hiện.
    target_class (int, tùy chọn): chỉ số của lớp dự đoán cần giải thích.
        Mặc định là lớp 0. Đặt None nếu mô hình là mô hình hồi quy.
    P (float, tùy chọn): xác suất để một cạnh tồn tại trong đồ thị ngẫu nhiên.
        Mặc định là mật độ của đồ thị gốc.
    deviation (float, tùy chọn): ngưỡng chênh lệch cho phép giữa tổng các giá trị Shapley
        và xác suất đầu ra dự đoán. Nếu là None, sẽ chạy đủ M bước lấy mẫu; nếu không,
        quá trình dừng khi đạt được độ xấp xỉ mong muốn (deviation). Tuy nhiên, sau M bước
        quá trình luôn kết thúc.
    log_odds (bool, tùy chọn): Mặc định là False. Nếu là True, dùng log odds thay cho
        xác suất làm giá trị để xấp xỉ Shapley.
    seed (float, tùy chọn): seed cho bộ sinh số ngẫu nhiên.

Trả về:
    list: danh sách các giá trị Shapley của các cạnh do EdgeSHAPer tính ra.
        Thứ tự các cạnh giống như trong self.edge_index.
"""

        if deviation is not None: # Nếu độ lệch không phải là None, gọi hàm explain_with_deviation.
            return self.explain_with_deviation(M = M, target_class = target_class, P = P, deviation = deviation, log_odds = log_odds, seed = seed, device=self.device)

        if target_class is None: # Nếu target_class là None, in ra thông báo và coi như mô hình là mô hình hồi quy.
            print("No target class specified. Regression model assumed.")

        E = self.edge_index # Lấy edge index.
        rng = default_rng(seed = seed) # Khởi tạo bộ sinh số ngẫu nhiên.
        self.model.eval() # Đặt mô hình ở chế độ đánh giá.
        phi_edges = [] # Danh sách các giá trị Shapley của các cạnh.

        num_nodes = self.x.shape[0] # Số nút trong đồ thị.
        num_edges = E.shape[1] # Số cạnh trong đồ thị.
        
        if P == None: # Nếu P là None, tính mật độ của đồ thị gốc.
            max_num_edges = num_nodes*(num_nodes-1) # Số cạnh tối đa trong đồ thị.
            graph_density = num_edges/max_num_edges # Mật độ của đồ thị gốc.
            P = graph_density # Đặt P bằng mật độ của đồ thị gốc.

        for j in tqdm(range(num_edges)): # Lặp qua từng cạnh trong đồ thị.
            marginal_contrib = 0 # Tổng đóng góp biên.
            for i in range(M): # Lặp M lần.
                E_z_mask = rng.binomial(1, P, num_edges) # Tạo mặt nạ nhị phân với xác suất P.
                E_mask = torch.ones(num_edges) # Tạo mặt nạ toàn 1.
                pi = torch.randperm(num_edges) # Tạo hoán vị ngẫu nhiên của chỉ số các cạnh.

                E_j_plus_index = torch.ones(num_edges, dtype=torch.int) # Tạo mặt nạ nhị phân cho tập hợp E_j+.
                E_j_minus_index = torch.ones(num_edges, dtype=torch.int) # Tạo mặt nạ nhị phân cho tập hợp E_j-.
                selected_edge_index = np.where(pi == j)[0].item() # Chỉ số của cạnh j trong hoán vị ngẫu nhiên pi.
                for k in range(num_edges): # Lặp qua các cạnh trong hoán vị ngẫu nhiên pi. 
                    if k <= selected_edge_index: # Nếu chỉ số của cạnh nhỏ hơn hoặc bằng chỉ số của cạnh j trong hoán vị ngẫu nhiên pi, 
                        E_j_plus_index[pi[k]] = E_mask[pi[k]] # Đặt giá trị của cạnh j trong tập hợp E_j+ bằng 1.
                    else:
                        E_j_plus_index[pi[k]] = E_z_mask[pi[k]] # Nếu không, đặt giá trị của cạnh j trong tập hợp E_j+ bằng E_z_mask[pi[k]] (mặt nạ nhị phân với xác suất P).

                for k in range(num_edges): # Lặp qua các cạnh trong hoán vị ngẫu nhiên pi. 
                    if k < selected_edge_index: # Nếu chỉ số của cạnh nhỏ hơn chỉ số của cạnh j trong hoán vị ngẫu nhiên pi, 
                        E_j_minus_index[pi[k]] = E_mask[pi[k]] # Đặt giá trị của cạnh j trong tập hợp E_j- bằng 1.
                    else:
                        E_j_minus_index[pi[k]] = E_z_mask[pi[k]] # Nếu không, đặt giá trị của cạnh j trong tập hợp E_j- bằng E_z_mask[pi[k]] (mặt nạ nhị phân với xác suất P).


                #we compute marginal contribs
                
                # with edge j # Tính toán với cạnh j 
                retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(self.device).squeeze() # Lấy chỉ số của các cạnh trong tập hợp E_j+.
                E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus) # Chọn các cạnh trong tập hợp E_j+.
                edge_weight_j_plus = None # Khởi tạo trọng số cạnh.

                if self.edge_weight is not None: # Nếu có trọng số cạnh,
                    edge_weight_j_plus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_plus) # Chọn các cạnh trong tập hợp E_j+.
                
                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device) # Tạo batch.
                
                out = self.model(self.x, E_j_plus, batch=batch, edge_weight=edge_weight_j_plus) # Tính toán đầu ra.
                out_prob = None # Khởi tạo xác suất đầu ra.
                V_j_plus = None # Khởi tạo giá trị Shapley của cạnh j.
                
                if target_class is not None: # Nếu có target class,
                    if not log_odds: # Nếu không dùng log odds,
                        out_prob = F.softmax(out, dim = 1) # Tính xác suất.
                    else:
                        out_prob = out # Nếu có log odds, out_prob sẽ chứa log_odds

                    V_j_plus = out_prob[0][target_class].item() # Lấy giá trị Shapley của cạnh j.
                else:
                    
                    out_prob = out # Nếu không có target class, out_prob sẽ chứa output của mô hình.
                
                    V_j_plus = out_prob[0][0].item() # Lấy giá trị Shapley của cạnh j.

                # without edge j # Tính toán mà không có cạnh j
                retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(self.device).squeeze() # Lấy chỉ số của các cạnh trong tập hợp E_j-.
                E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus) # Chọn các cạnh trong tập hợp E_j-.
                edge_weight_j_minus = None # Khởi tạo trọng số cạnh.

                if self.edge_weight is not None: # Nếu có trọng số cạnh,    
                    edge_weight_j_minus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_minus) # Chọn các cạnh trong tập hợp E_j-.

                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device) # Tạo batch.
                out = self.model(self.x, E_j_minus, batch=batch, edge_weight=edge_weight_j_minus) # Tính toán đầu ra.

                V_j_minus = None # Khởi tạo giá trị Shapley của cạnh j.
                
                if target_class is not None: # Nếu có target class,
                    if not log_odds: # Nếu không dùng log odds,
                        out_prob = F.softmax(out, dim = 1) # Tính xác suất.
                    else: # Nếu có log odds,
                        out_prob = out     #out prob variable now containts log_odds

                    V_j_minus = out_prob[0][target_class].item() # Lấy giá trị Shapley của cạnh j.
                else:
                    out_prob = out # Nếu không có target class, out_prob sẽ chứa output của mô hình.
                
                    V_j_minus = out_prob[0][0].item() # Lấy giá trị Shapley của cạnh j.

                marginal_contrib += (V_j_plus - V_j_minus) # Tính toán đóng góp cận biên.

            phi_edges.append(marginal_contrib/M) # Thêm giá trị Shapley của cạnh j vào danh sách.

        self.target_class = target_class # Đặt target class.
        self.explained = True    # Đặt explained = True.
        self.phi_edges = phi_edges # Đặt phi_edges = giá trị Shapley của cạnh j.
        return phi_edges # Trả về giá trị Shapley của cạnh j.


    def explain_with_deviation(self, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = None, device = "cpu"): # Giải thích với độ lệch.   
        
        if target_class is None: # Nếu không có target class,
            print("No target class specified. Regression model assumed.") # In thông báo.
            
        rng = default_rng(seed = seed) # Tạo rng với seed.
        self.model.eval() # Đặt mô hình ở chế độ đánh giá.
        batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device) # Tạo batch.
        E = self.edge_index # Lấy edge_index.
        out = self.model(self.x, E, batch=batch, edge_weight=self.edge_weight) # Tính toán đầu ra.
        out_prob_real = F.softmax(out, dim = 1)[0][target_class].item() if target_class is not None else out[0][0].item() # Tính xác suất đầu ra.

        num_nodes = self.x.shape[0] # Số lượng nút.
        num_edges = E.shape[1] # Số lượng cạnh.

        phi_edges = [] # Khởi tạo danh sách giá trị Shapley.
        phi_edges_current = [0] * num_edges # Khởi tạo danh sách giá trị Shapley hiện tại.
        
        if P == None: # Nếu không có P,
            max_num_edges = num_nodes*(num_nodes-1) # Số lượng cạnh tối đa.
            graph_density = num_edges/max_num_edges # Mật độ đồ thị.
            P = graph_density # Đặt P = mật độ đồ thị.  

        for i in tqdm(range(M)): # Thực hiện M lần lặp.
        
            for j in range(num_edges): # Thực hiện num_edges lần lặp.
                
                
                E_z_mask = rng.binomial(1, P, num_edges) # Tạo E_z_mask.
                E_mask = torch.ones(num_edges) # Tạo E_mask.
                pi = torch.randperm(num_edges) # Tạo pi.

                E_j_plus_index = torch.ones(num_edges, dtype=torch.int) # Tạo E_j_plus_index.
                E_j_minus_index = torch.ones(num_edges, dtype=torch.int) # Tạo E_j_minus_index.
                selected_edge_index = np.where(pi == j)[0].item() # Lấy chỉ số của cạnh j.
                for k in range(num_edges): # Thực hiện num_edges lần lặp.
                    if k <= selected_edge_index: # Nếu k <= selected_edge_index,
                        E_j_plus_index[pi[k]] = E_mask[pi[k]] # Đặt E_j_plus_index[pi[k]] = E_mask[pi[k]].
                    else: # Nếu k > selected_edge_index,
                        E_j_plus_index[pi[k]] = E_z_mask[pi[k]] # Đặt E_j_plus_index[pi[k]] = E_z_mask[pi[k]].

                for k in range(num_edges): # Thực hiện num_edges lần lặp.
                    if k < selected_edge_index: # Nếu k < selected_edge_index,
                        E_j_minus_index[pi[k]] = E_mask[pi[k]] # Đặt E_j_minus_index[pi[k]] = E_mask[pi[k]].
                    else: # Nếu k >= selected_edge_index,
                        E_j_minus_index[pi[k]] = E_z_mask[pi[k]] # Đặt E_j_minus_index[pi[k]] = E_z_mask[pi[k]].


                #we compute marginal contribs
                
                # with edge j #với cạnh j
                retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze() # Lấy chỉ số của các cạnh trong tập hợp E_j+.
                E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus) # Chọn các cạnh trong tập hợp E_j+.
                edge_weight_j_plus = None # Khởi tạo trọng số cạnh.

                if self.edge_weight is not None: # Nếu có trọng số cạnh,
                    edge_weight_j_plus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_plus) # Chọn các cạnh trong tập hợp E_j+.

                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device) # Tạo batch.
                
                out = self.model(self.x, E_j_plus, batch=batch, edge_weight=edge_weight_j_plus) # Tính toán đầu ra.
                out_prob = None # Khởi tạo xác suất đầu ra.

                V_j_plus = None # Khởi tạo giá trị Shapley của cạnh j.
                
                if target_class is not None: # Nếu có target class,
                    if not log_odds: # Nếu không dùng log odds,
                        out_prob = F.softmax(out, dim = 1) # Tính xác suất.
                    else: # Nếu có log odds,
                        out_prob = out #out prob variable now containts log_odds

                    V_j_plus = out_prob[0][target_class].item() # Lấy giá trị Shapley của cạnh j.
                else:
                    
                    out_prob = out #out prob variable now containts log_odds
                
                    V_j_plus = out_prob[0][0].item()

                # without edge j #không có cạnh j
                retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze() # Lấy chỉ số của các cạnh trong tập hợp E_j-.
                E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus) # Chọn các cạnh trong tập hợp E_j-.
                edge_weight_j_minus = None # Khởi tạo trọng số cạnh.

                if self.edge_weight is not None: # Nếu có trọng số cạnh,
                    edge_weight_j_minus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_minus) # Chọn các cạnh trong tập hợp E_j-.

                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device) # Tạo batch.
                out = self.model(self.x, E_j_minus, batch=batch, edge_weight=edge_weight_j_minus) # Tính toán đầu ra.

                V_j_minus = None # Khởi tạo giá trị Shapley của cạnh j.
                
                if target_class is not None: # Nếu có target class,
                    if not log_odds: # Nếu không dùng log odds,
                        out_prob = F.softmax(out, dim = 1) # Tính xác suất.
                    else: # Nếu có log odds,
                        out_prob = out #out prob variable now containts log_odds

                    V_j_minus = out_prob[0][target_class].item() # Lấy giá trị Shapley của cạnh j.
                else:
                    out_prob = out #out prob variable now containts log_odds
                
                    V_j_minus = out_prob[0][0].item() # Lấy giá trị Shapley của cạnh j.

                phi_edges_current[j] += (V_j_plus - V_j_minus) # Cộng dồn giá trị Shapley.

            
            phi_edges = [elem / (i+1) for elem in phi_edges_current] # Tính trung bình giá trị Shapley.
            # print(sum(phi_edges))
            if abs(out_prob_real - sum(phi_edges)) <= deviation: # Nếu giá trị tuyệt đối của sự khác biệt giữa xác suất dự đoán ban đầu và tổng giá trị Shapley nhỏ hơn độ lệch,
                break

        self.phi_edges = phi_edges # Lưu giá trị Shapley.   
        self.target_class = target_class # Lưu target class.
        self.explained = True    # Đặt cờ đã giải thích là True.
        return phi_edges # Trả về giá trị Shapley.

    def compute_original_predicted_probability(self): # Tính xác suất dự đoán ban đầu.
        self.model.eval() # Đặt mô hình ở chế độ đánh giá.
        batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device) # Tạo batch.
        out_log_odds = self.model(self.x, self.edge_index, batch=batch, edge_weight=self.edge_weight) # Tính toán đầu ra.
        out_prob = F.softmax(out_log_odds, dim = 1) # Tính xác suất.
        original_pred_prob = out_prob[0][self.target_class].item() # Lấy xác suất dự đoán ban đầu.

        self.original_pred_prob = original_pred_prob # Lưu xác suất dự đoán ban đầu.

    def compute_pertinent_positive_set(self, verbose = False): # Tính tập hợp dương.
        assert(self.explained) # Đảm bảo rằng giải thích đã được tính toán.
        
        if self.target_class is None: # Nếu không có target class,
            raise Exception("Minimal informative sets are not defined for regression problems.") # Ném ngoại lệ.

        if self.original_pred_prob is None: # Nếu không có xác suất dự đoán ban đầu,
            self.compute_original_predicted_probability() # Tính xác suất dự đoán ban đầu.

        self.model.eval() # Đặt mô hình ở chế độ đánh giá.
        infidelity = 1 # None it was none, last edit since it remains none if the class does not change, so we put 1
        important_edges_ranking = np.argsort(-np.array(self.phi_edges)) # Sắp xếp các cạnh theo giá trị Shapley giảm dần.
        for i in range(1, important_edges_ranking.shape[0]+1): # Lặp qua các cạnh theo thứ tự giảm dần của giá trị Shapley.
            reduced_edge_index = torch.index_select(self.edge_index, dim = 1, index = torch.LongTensor(important_edges_ranking[0:i]).to(self.device)) # Chọn các cạnh trong tập hợp E_j+.
            batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device) # Tạo batch.
            out = self.model(self.x, reduced_edge_index, batch=batch) # Tính toán đầu ra.
            out_prob = F.softmax(out, dim = 1) # Tính xác suất.
            # print(out_prob)
            predicted_class = torch.argmax(out_prob[0]).item() # Lấy lớp dự đoán.
            if (predicted_class == self.target_class): # Nếu lớp dự đoán bằng lớp mục tiêu,
                
                
                pred_prob = out_prob[0][self.target_class].item() # Lấy xác suất dự đoán.
                infidelity = self.original_pred_prob-pred_prob # Tính độ trung thực.

                if verbose: # Nếu verbose là True,
                    print("FID- using pertinent positive set: ", infidelity) # In độ trung thực.
                break

        self.pertinent_positive_set = reduced_edge_index # Lưu tập hợp dương.
        self.infidelity = infidelity # Lưu độ trung thực.
        return reduced_edge_index, infidelity # Trả về tập hợp dương và độ trung thực.

    #for legacy with old code, we keep the mispelled function which is now deprecated but calls the correct one
    def compute_pertinent_positivite_set(self, verbose = False): # Hàm tính tập hợp dương (từ tên cũ, đã được đổi tên thành compute_pertinent_positive_set).
        print("WARNING: compute_pertinent_positivite_set is now deprecated, use compute_pertinent_positive_set instead.")
        return self.compute_pertinent_positive_set(verbose)

    def compute_minimal_top_k_set(self, verbose = False): # Hàm tính tập hợp cực tiểu Top-k.
        assert(self.explained) # Đảm bảo rằng giải thích đã được tính toán.
        
        if self.target_class is None: # Nếu không có target class,
            raise Exception("Minimal informative sets are not defined for regression problems.") # Ném ngoại lệ.
            
        if self.original_pred_prob is None: # Nếu không có xác suất dự đoán ban đầu,
            self.compute_original_predicted_probability() # Tính xác suất dự đoán ban đầu.

        self.model.eval() # Đặt mô hình ở chế độ đánh giá.
        fidelity = 0 # Khởi tạo độ trung thực.
        pertinent_set_indices = [] # Khởi tạo danh sách chỉ số của các cạnh trong tập hợp dương.
        pertinent_set_edge_index = None # Khởi tạo chỉ số của các cạnh trong tập hợp dương.
        important_edges_ranking = np.argsort(-np.array(self.phi_edges)) # Sắp xếp các cạnh theo giá trị Shapley giảm dần.
        for i in range(important_edges_ranking.shape[0]): # Lặp qua các cạnh theo thứ tự giảm dần của giá trị Shapley.
            index_of_edge_to_remove = important_edges_ranking[i] # Lấy chỉ số của cạnh cần loại bỏ.
            pertinent_set_indices.append(index_of_edge_to_remove) # Thêm chỉ số của cạnh vào danh sách.

            reduced_edge_index = torch.index_select(self.edge_index, dim = 1, index = torch.LongTensor(important_edges_ranking[i:]).to(self.device)) # Chọn các cạnh trong tập hợp E_j+.
            
            # all nodes belong to same graph
            batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device) # Tạo batch.
            out = self.model(self.x, reduced_edge_index, batch=batch) # Tính toán đầu ra.
            out_prob = F.softmax(out, dim = 1) # Tính xác suất.
            # print(out_prob)
            predicted_class = torch.argmax(out_prob[0]).item() # Lấy lớp dự đoán.

            if predicted_class != self.target_class: # Nếu lớp dự đoán không bằng lớp mục tiêu,
                pred_prob = out_prob[0][self.target_class].item() # Lấy xác suất dự đoán.
                fidelity = self.original_pred_prob - pred_prob # Tính độ trung thực.
                if verbose: # Nếu verbose là True,
                    print("FID+ using minimal top-k set: ", fidelity) # In độ trung thực.
                break

        pertinent_set_edge_index = torch.index_select(self.edge_index, dim = 1, index = torch.LongTensor(pertinent_set_indices).to(self.device)) # Chọn các cạnh trong tập hợp E_j+.
        
        self.minimal_top_k_set = pertinent_set_edge_index # Lưu tập hợp cực tiểu Top-k.
        self.fidelity = fidelity # Lưu độ trung thực.

        return pertinent_set_edge_index, fidelity # Trả về tập hợp cực tiểu Top-k và độ trung thực.


    def compute_trustworthiness(self, verbose = False): # Hàm tính độ tin cậy.
        ''' 
        Computes the Trustworthiness (TW) of the explanation. The TW metric is defined as an harmonic mean of the fidelity and reverse of the infidelity.
        '''

        assert(self.explained) # Đảm bảo rằng giải thích đã được tính toán.

        assert(self.fidelity is not None) # Đảm bảo rằng độ trung thực đã được tính toán.
        assert(self.infidelity is not None) # Đảm bảo rằng độ trung thực ngược đã được tính toán.

        TW = None
        
        if self.fidelity+(1-self.infidelity) == 0: # Nếu mẫu số bằng 0,
            TW = 0 # Đặt độ tin cậy bằng 0.
        else:
            TW = 2* ((self.fidelity*(1-self.infidelity))/(self.fidelity+(1-self.infidelity))) # Tính độ tin cậy.

        self.trustworthiness = TW # Lưu độ tin cậy.

        if verbose: # Nếu verbose là True,
            print("Trustworthiness: ", self.trustuworthiness) # In độ tin cậy.

        return self.trustworthiness # Trả về độ tin cậy.

    def visualize_molecule_explanations(self, smiles, save_path=None, pertinent_positive = False, minimal_top_k = False): # Hàm vẽ sơ đồ giải thích phân tử.
        assert(self.explained) # Đảm bảo rằng giải thích đã được tính toán.

        img_expl = None # Ảnh giải thích.
        img_pert_pos = None # Ảnh tập hợp dương.
        img_min_top_k = None # Ảnh tập hợp cực tiểu Top-k.

        edge_index = self.edge_index.to("cpu") # Lấy chỉ số của các cạnh.

        test_mol = Chem.MolFromSmiles(smiles) # Tạo phân tử.
        test_mol = Draw.PrepareMolForDrawing(test_mol) # Chuẩn bị phân tử cho việc vẽ.

        num_bonds = len(test_mol.GetBonds()) # Số liên kết.

        rdkit_bonds = {} # Từ điển các liên kết.

        for i in range(num_bonds): # Lặp qua các liên kết.
            init_atom = test_mol.GetBondWithIdx(i).GetBeginAtomIdx() # Lấy nguyên tử bắt đầu.
            end_atom = test_mol.GetBondWithIdx(i).GetEndAtomIdx() # Lấy nguyên tử kết thúc.
            
            rdkit_bonds[(init_atom, end_atom)] = i # Lưu liên kết vào từ điển.

        rdkit_bonds_phi = [0]*num_bonds # Danh sách các giá trị Shapley cho mỗi liên kết.
        for i in range(len(self.phi_edges)): # Lặp qua các giá trị Shapley.
            phi_value = self.phi_edges[i] # Lấy giá trị Shapley.
            init_atom = edge_index[0][i].item() # Lấy nguyên tử bắt đầu.
            end_atom = edge_index[1][i].item() # Lấy nguyên tử kết thúc.
            
            if (init_atom, end_atom) in rdkit_bonds: # Kiểm tra xem có trong từ điển.
                bond_index = rdkit_bonds[(init_atom, end_atom)] # Lấy chỉ số của liên kết.
                rdkit_bonds_phi[bond_index] += phi_value # Cộng giá trị Shapley vào danh sách.
            if (end_atom, init_atom) in rdkit_bonds: # Kiểm tra xem có trong từ điển.
                bond_index = rdkit_bonds[(end_atom, init_atom)] # Lấy chỉ số của liên kết.
                rdkit_bonds_phi[bond_index] += phi_value # Cộng giá trị Shapley vào danh sách.

        plt.clf() # Xóa đồ thị.
        canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi, atom_width=0.2, bond_length=0.5, bond_width=0.5) # Ánh xạ giá trị vào phân tử.
        img_expl = transform2png(canvas.GetDrawingText()) # Chuyển đổi sang định dạng PNG.

        if save_path is not None: # Nếu có đường dẫn lưu,
            img_expl.save(save_path + "/" + "EdgeSHAPer_explanations_heatmap.png", dpi = (300,300)) # Lưu ảnh.

        if pertinent_positive: # Nếu có tập hợp dương,
            if self.pertinent_positive_set is None: # Nếu chưa tính toán, tính toán.
                self.compute_pertinent_positivite_set()

            rdkit_bonds_phi_pertinent_set = [0]*num_bonds # Danh sách các giá trị Shapley cho mỗi liên kết.
            pertinent_set_edge_index = self.pertinent_positive_set # Lấy tập hợp cực tiểu.
            
            for i in range(pertinent_set_edge_index.shape[1]): # Lặp qua các cạnh.
                
                init_atom = pertinent_set_edge_index[0][i].item() # Lấy nguyên tử bắt đầu.
                end_atom = pertinent_set_edge_index[1][i].item() # Lấy nguyên tử kết thúc.
                
                
                if (init_atom, end_atom) in rdkit_bonds: # Kiểm tra xem có trong từ điển.
                    bond_index = rdkit_bonds[(init_atom, end_atom)] # Lấy chỉ số của liên kết.
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0: # Kiểm tra xem có trong danh sách.
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index] # Cộng giá trị Shapley vào danh sách.
                if (end_atom, init_atom) in rdkit_bonds: # Kiểm tra xem có trong từ điển.
                    bond_index = rdkit_bonds[(end_atom, init_atom)] # Lấy chỉ số của liên kết.
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0: # Kiểm tra xem có trong danh sách.
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index] # Cộng giá trị Shapley vào danh sách.

            plt.clf()    # Xóa đồ thị.
            canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi_pertinent_set, atom_width=0.2, bond_length=0.5, bond_width=0.5) # Ánh xạ giá trị vào phân tử.
            img_pert_pos = transform2png(canvas.GetDrawingText()) # Chuyển đổi sang định dạng PNG.

            if save_path is not None: # Nếu có đường dẫn lưu,

                img_pert_pos.save(save_path + "/" + "EdgeSHAPer_pertinent_positive_set_heatmap.png", dpi = (300,300)) # Lưu ảnh.

        if minimal_top_k: # Nếu có tập hợp cực tiểu,
            if self.minimal_top_k_set is None: # Nếu chưa tính toán, tính toán.
                self.compute_minimal_top_k_set() # Tính toán tập hợp cực tiểu.

            rdkit_bonds_phi_pertinent_set = [0]*num_bonds # Danh sách các giá trị Shapley cho mỗi liên kết.
            pertinent_set_edge_index = self.minimal_top_k_set # Lấy tập hợp cực tiểu.
            for i in range(pertinent_set_edge_index.shape[1]): # Lặp qua các cạnh.
                
                init_atom = pertinent_set_edge_index[0][i].item() # Lấy nguyên tử bắt đầu.
                end_atom = pertinent_set_edge_index[1][i].item() # Lấy nguyên tử kết thúc.
                
                
                if (init_atom, end_atom) in rdkit_bonds: # Kiểm tra xem có trong từ điển.
                    bond_index = rdkit_bonds[(init_atom, end_atom)] # Lấy chỉ số của liên kết.
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0: # Kiểm tra xem có trong danh sách.
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index] # Cộng giá trị Shapley vào danh sách.
                if (end_atom, init_atom) in rdkit_bonds: # Kiểm tra xem có trong từ điển.
                    bond_index = rdkit_bonds[(end_atom, init_atom)] # Lấy chỉ số của liên kết.
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0: # Kiểm tra xem có trong danh sách.
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index] # Cộng giá trị Shapley vào danh sách.

            plt.clf() # Xóa đồ thị.
            canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi_pertinent_set, atom_width=0.2, bond_length=0.5, bond_width=0.5) # Ánh xạ giá trị vào phân tử.
            img_min_top_k = transform2png(canvas.GetDrawingText()) # Chuyển đổi sang định dạng PNG.

            if save_path is not None: # Nếu có đường dẫn lưu,
                img_min_top_k.save(save_path + "/" + "EdgeSHAPer_minimal_top_k_set_heatmap.png", dpi = (300,300)) # Lưu ảnh.

        plt.clf()
        return img_expl, img_pert_pos, img_min_top_k    # Trả về ảnh giải thích, ảnh tập hợp cực tiểu và ảnh tập hợp cực tiểu có gộp các liên kết có giá trị nhỏ nhất.


    


###EdgeSHAPer as a function###
def edgeshaper(model, x, E, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = 42, edge_weight = None, device = "cpu"): # Tính toán xấp xỉ giá trị Shapley cho mỗi liên kết.
   """
Tính xấp xỉ giá trị Shapley cho mức độ quan trọng của các cạnh trong GNN.

Tham số:
    model (Torch.NN.Module): mô hình GNN của PyTorch được sử dụng.
    x (tensor): tensor 2 chiều chứa đặc trưng của các nút trong đồ thị cần giải thích.
    E (tensor): edge index của đồ thị cần giải thích.
    P (float, tùy chọn): xác suất để một cạnh tồn tại trong đồ thị ngẫu nhiên.
        Mặc định là mật độ của đồ thị gốc.
    M (int): số bước lấy mẫu Monte Carlo cần thực hiện.
    target_class (int, tùy chọn): chỉ số của lớp dự đoán cần giải thích.
        Mặc định là lớp 0.
    deviation (float, tùy chọn): ngưỡng chênh lệch cho phép giữa tổng các giá trị Shapley
        và xác suất đầu ra dự đoán. Nếu là None, sẽ chạy đủ M bước lấy mẫu; nếu không,
        quá trình dừng khi đạt được độ xấp xỉ mong muốn (deviation). Tuy nhiên, sau M bước
        quá trình luôn kết thúc.
    log_odds (bool, tùy chọn): Mặc định là False. Nếu là True, dùng log odds thay cho
        xác suất làm giá trị để xấp xỉ Shapley.
    seed (float, tùy chọn): seed cho bộ sinh số ngẫu nhiên.
    device (string, tùy chọn): cho biết dùng cuda hay cpu.

Trả về:
    list: danh sách các giá trị Shapley của các cạnh do EdgeSHAPer tính ra.
        Thứ tự các cạnh giống như trong E.
"""

    if deviation != None: # Nếu có ngưỡng chênh lệch,
        return edgeshaper_deviation(model, x, E, M = M, target_class = target_class, P = P, deviation = deviation, log_odds = log_odds, seed = seed, edge_weight = edge_weight, device=device) # Gọi hàm tính toán xấp xỉ giá trị Shapley.


    rng = default_rng(seed = seed) # Khởi tạo bộ sinh số ngẫu nhiên.
    model.eval() # Đặt mô hình vào chế độ đánh giá.
    phi_edges = [] # Danh sách các giá trị Shapley cho mỗi liên kết.

    num_nodes = x.shape[0] # Số lượng nút.
    num_edges = E.shape[1] # Số lượng cạnh.
    
    if P == None: # Nếu không có xác suất, tính toán mật độ đồ thị.
        max_num_edges = num_nodes*(num_nodes-1) # Số lượng cạnh tối đa. 
        graph_density = num_edges/max_num_edges # Mật độ đồ thị.
        P = graph_density # Xác suất.

    for j in tqdm(range(num_edges)): # Lặp qua các cạnh.
        marginal_contrib = 0 # Khởi tạo đóng góp biên.
        for i in range(M): # Lặp M lần.
            E_z_mask = rng.binomial(1, P, num_edges) # Tạo mask nhị phân cho tập hợp ngẫu nhiên Z.
            E_mask = torch.ones(num_edges) # Tạo mask nhị phân cho tập hợp đầy đủ.
            pi = torch.randperm(num_edges) # Tạo hoán vị ngẫu nhiên của các chỉ số cạnh.

            E_j_plus_index = torch.ones(num_edges, dtype=torch.int) # Tạo mask nhị phân cho tập hợp S U {j}.    
            E_j_minus_index = torch.ones(num_edges, dtype=torch.int) # Tạo mask nhị phân cho tập hợp S.
            selected_edge_index = np.where(pi == j)[0].item() # Vị trí của cạnh j trong hoán vị ngẫu nhiên.
            for k in range(num_edges): # Lặp qua các cạnh.
                if k <= selected_edge_index: # Nếu k nhỏ hơn hoặc bằng vị trí của cạnh j trong hoán vị ngẫu nhiên,
                    E_j_plus_index[pi[k]] = E_mask[pi[k]] # Gán giá trị của mask đầy đủ vào mask của tập hợp S U {j}.
                else:
                    E_j_plus_index[pi[k]] = E_z_mask[pi[k]] # Gán giá trị của mask ngẫu nhiên vào mask của tập hợp S U {j}.

            for k in range(num_edges): # Lặp qua các cạnh.
                if k < selected_edge_index: # Nếu k nhỏ hơn vị trí của cạnh j trong hoán vị ngẫu nhiên,
                    E_j_minus_index[pi[k]] = E_mask[pi[k]] # Gán giá trị của mask đầy đủ vào mask của tập hợp S.
                else:
                    E_j_minus_index[pi[k]] = E_z_mask[pi[k]] # Gán giá trị của mask ngẫu nhiên vào mask của tập hợp S.


            #we compute marginal contribs
            
            # with edge j # với cạnh j
            retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze() # Lấy các chỉ số của các cạnh trong tập hợp S U {j}.
            E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus) # Chọn các cạnh trong tập hợp S U {j}.
            edge_weight_j_plus = None # Khởi tạo trọng số của cạnh j.

            if edge_weight is not None: # Nếu có trọng số của cạnh,
                edge_weight_j_plus = torch.index_select(edge_weight, dim = 0, index = retained_indices_plus) # Chọn trọng số của các cạnh trong tập hợp S U {j}.

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device) # Batch size bằng số lượng nút.
            
            out = model(x, E_j_plus, batch=batch, edge_weight = edge_weight_j_plus) # Truyền vào mô hình.
            out_prob = 0 # Khởi tạo xác suất.

            V_j_plus = 0 # Khởi tạo giá trị dự đoán của lớp.
                
            if target_class is not None: # Nếu có lớp mục tiêu,
                if not log_odds: # Nếu không có log odds,
                    out_prob = F.softmax(out, dim = 1)
                else:
                    out_prob = out #out prob variable now containts log_odds

                V_j_plus = out_prob[0][target_class].item() 
            else: # Nếu không có lớp mục tiêu,
                
                out_prob = out # out_prob variable now containts log_odds
            
                V_j_plus = out_prob[0][0].item()

            # without edge j # với cạnh j
            retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze() # Lấy các chỉ số của các cạnh trong tập hợp S.
            E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus) # Chọn các cạnh trong tập hợp S.
            edge_weight_j_minus = None # Khởi tạo trọng số của cạnh j.

            if edge_weight is not None: # Nếu có trọng số của cạnh,
                edge_weight_j_minus = torch.index_select(edge_weight, dim = 0, index = retained_indices_minus) # Chọn trọng số của các cạnh trong tập hợp S.

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device) # Batch size bằng số lượng nút.
            out = model(x, E_j_minus, batch=batch, edge_weight = edge_weight_j_minus) # Truyền vào mô hình.

            V_j_minus = 0 # Khởi tạo giá trị dự đoán của lớp.
                
            if target_class is not None: # Nếu có lớp mục tiêu,
                if not log_odds: # Nếu không có log odds,
                    out_prob = F.softmax(out, dim = 1) # Tính xác suất.
                else:
                    out_prob = out #out prob variable now containts log_odds

                V_j_minus = out_prob[0][target_class].item() # Lấy giá trị dự đoán.
            else:
                out_prob = out # out_prob variable now containts log_odds
            
                V_j_minus = out_prob[0][0].item() # Lấy giá trị dự đoán.

            marginal_contrib += (V_j_plus - V_j_minus) # Tính toán đóng góp biên.

        phi_edges.append(marginal_contrib/M) # Tính toán giá trị trung bình của đóng góp biên.
        
    return phi_edges # Trả về đóng góp biên của mỗi cạnh.


def edgeshaper_deviation(model, x, E, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = 42, edge_weight = None, device = "cpu"): # Tính toán độ lệch của đóng góp biên của mỗi cạnh so với giá trị trung bình.

    rng = default_rng(seed = seed) # Khởi tạo rng với seed.
    model.eval() # Đặt mô hình ở chế độ đánh giá.
    batch = torch.zeros(x.shape[0], dtype=int, device=x.device) # Tạo batch size bằng số lượng nút.
    out = model(x, E, batch=batch, edge_weight = edge_weight) # Truyền vào mô hình.
    out_prob_real = F.softmax(out, dim = 1)[0][target_class].item() if target_class is not None else out[0][0].item() # Tính toán xác suất dự đoán.

    num_nodes = x.shape[0] # Lấy số lượng nút.
    num_edges = E.shape[1] # Lấy số lượng cạnh.

    phi_edges = [] # Danh sách lưu trữ đóng góp biên.
    phi_edges_current = [0] * num_edges # Tạo danh sách lưu trữ đóng góp biên.
    
    if P == None: # Nếu không có P,
        max_num_edges = num_nodes*(num_nodes-1) # Tính số lượng cạnh tối đa.
        graph_density = num_edges/max_num_edges # Tính mật độ đồ thị.
        P = graph_density # Gán giá trị của P bằng mật độ đồ thị.

    for i in tqdm(range(M)): # Lặp lại M lần.
    
        for j in range(num_edges): # Lặp lại số lượng cạnh.
            
            
            E_z_mask = rng.binomial(1, P, num_edges) # Tạo mask ngẫu nhiên.
            E_mask = torch.ones(num_edges) # Tạo mask đầy đủ.
            pi = torch.randperm(num_edges) # Tạo hoán vị ngẫu nhiên của các cạnh.

            E_j_plus_index = torch.ones(num_edges, dtype=torch.int) # Tạo danh sách lưu trữ các cạnh trong tập hợp S U {j}.
            E_j_minus_index = torch.ones(num_edges, dtype=torch.int) # Tạo danh sách lưu trữ các cạnh trong tập hợp S.
            selected_edge_index = np.where(pi == j)[0].item() # Lấy vị trí của cạnh j trong hoán vị ngẫu nhiên.
            for k in range(num_edges): # Lặp lại số lượng cạnh.
                if k <= selected_edge_index: # Nếu k nhỏ hơn hoặc bằng vị trí của cạnh j trong hoán vị ngẫu nhiên,
                    E_j_plus_index[pi[k]] = E_mask[pi[k]] # Gán giá trị của mask đầy đủ vào mask của tập hợp S.
                else:
                    E_j_plus_index[pi[k]] = E_z_mask[pi[k]] # Gán giá trị của mask ngẫu nhiên vào mask của tập hợp S.

            for k in range(num_edges): # Lặp lại số lượng cạnh.
                if k < selected_edge_index: # Nếu k nhỏ hơn vị trí của cạnh j trong hoán vị ngẫu nhiên,
                    E_j_minus_index[pi[k]] = E_mask[pi[k]] # Gán giá trị của mask đầy đủ vào mask của tập hợp S.
                else:
                    E_j_minus_index[pi[k]] = E_z_mask[pi[k]] # Gán giá trị của mask ngẫu nhiên vào mask của tập hợp S.


            #we compute marginal contribs
            
            # with edge j #với cạnh j
            retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze() # Lấy các chỉ số của các cạnh trong tập hợp S U {j}.
            E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus) # Chọn các cạnh trong tập hợp S.

            edge_weight_j_plus = None # Khởi tạo trọng số của cạnh j.

            if edge_weight is not None: # Nếu có trọng số của cạnh,
                edge_weight_j_plus = torch.index_select(edge_weight, dim = 0, index = retained_indices_plus) # Chọn trọng số của các cạnh trong tập hợp S.

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device) # Batch size bằng số lượng nút.
            
            out = model(x, E_j_plus, batch=batch, edge_weight = edge_weight_j_plus) # Truyền vào mô hình.   
            out_prob = None # Khởi tạo xác suất.    

            V_j_plus = None # Khởi tạo giá trị dự đoán.
                
            if target_class is not None: # Nếu có lớp mục tiêu,
                if not log_odds: # Nếu không có log odds,
                    out_prob = F.softmax(out, dim = 1) # Tính xác suất.
                else: # Nếu có log odds,
                    out_prob = out #out prob variable now containts log_odds

                V_j_plus = out_prob[0][target_class].item() # Lấy giá trị dự đoán.
            else:
                out_prob = out #out prob variable now containts log_odds
            
                V_j_plus = out_prob[0][0].item() # Lấy giá trị dự đoán.

            # without edge j #không có cạnh j
            retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze() # Lấy các chỉ số của các cạnh trong tập hợp S U {j}.
            E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus) # Chọn các cạnh trong tập hợp S.

            edge_weight_j_minus = None # Khởi tạo trọng số của cạnh j.

            if edge_weight is not None: # Nếu có trọng số của cạnh,
                edge_weight_j_minus = torch.index_select(edge_weight, dim = 0, index = retained_indices_minus) # Chọn trọng số của các cạnh trong tập hợp S.

            batch = torch.zeros(x.shape[0], dtype=int, device=x.device) # Batch size bằng số lượng nút.
            out = model(x, E_j_minus, batch=batch, edge_weight = edge_weight_j_minus) # Truyền vào mô hình.

            V_j_minus = None # Khởi tạo giá trị dự đoán.
                
            if target_class is not None: # Nếu có lớp mục tiêu,
                if not log_odds: # Nếu không có log odds,
                    out_prob = F.softmax(out, dim = 1) # Tính xác suất.
                    out_prob = F.softmax(out, dim = 1)
                else:
                    out_prob = out #out prob variable now containts log_odds

                V_j_minus = out_prob[0][target_class].item() # Lấy giá trị dự đoán.
            else:
                out_prob = out #out prob variable now containts log_odds
            
                V_j_minus = out_prob[0][0].item() # Lấy giá trị dự đoán.

            phi_edges_current[j] += (V_j_minus - V_j_minus) # Tính đóng góp biên.

        
        phi_edges = [elem / (i+1) for elem in phi_edges_current] # Tính đóng góp biên trung bình.
        # print(sum(phi_edges))
        if abs(out_prob_real - sum(phi_edges)) <= deviation: # Nếu độ lệch nhỏ hơn deviation, break.
            break
             
    return phi_edges # Trả về đóng góp biên của mỗi cạnh.