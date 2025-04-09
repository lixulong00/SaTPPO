import gym
import numpy as np

# Relevant parameters of UAV flight energy consumption
V_m = 16.67
P_bla = 1. / 8. * 0.012 * 1.225 * 0.05 * 0.503 * (100 ** 3) * (0.4 ** 3)
P_ind = (1 + 0.1) * (20 ** (3. / 2.)) / np.sqrt(2 * 1.225 * 0.503)
term1 = P_bla * (1. + (3. * V_m * V_m) / (120 * 120))
term2 = P_ind * ((np.sqrt(1 + ((V_m ** 4) / (4 * 4.03 * 4.03))) - ((V_m * V_m) / (2 * 4.03 * 4.03))) ** (1. / 2.))
term3 = (1. / 2.) * 0.6 * 1.225 * 0.05 * 0.503 * (V_m ** 3)
T_n_hover = 0.01

# Load simulation data sets that are randomly generated for code runnability. Please replace the simulation data set with the real data set, and further adjust the relevant parameters for better optimization results.
Task_all = np.load('SimulationDdata1.npy', allow_pickle=True)
Task_size_max, Task_size_min = Task_all[:, :, :, 0].max(), Task_all[:, :, :, 0].min()
Task_cpu_max, Task_cpu_min = Task_all[:, :, :, 1].max(), Task_all[:, :, :, 1].min()
Task_delay_max, Task_delay_min = Task_all[:, :, :, 2].max(), Task_all[:, :, :, 2].min()

class env(gym.Env):
    def __init__(self, args):
        args.IoT_tra_power_max = 1.0
        self.args = args
        self.args.W = self.args.W * 1e6
        self.args.CPU_F_U = self.args.CPU_F_U * 1e9
        self.args.N = 1
        self.UAV_n = self.args.N
        self.IoT_n = self.args.M
        self.Task_type_n = self.args.Task_type_n
        self.UAV_xyz_init = np.array([[0.,0.,50.]])
        self.IoT_tra_power = np.ones((self.IoT_n,)) * args.IoT_tra_power_max
        self.Task = np.zeros((self.IoT_n, 3))
        
        self.action_S_dim =  2 * (self.Task_type_n + 1) * self.UAV_n
        self.action_U_dim = 3*self.UAV_n + 2*self.IoT_n
        self.state_dim =  1 + self.Task_type_n + self.Task_type_n * 2 + self.UAV_n * 4 + 8 * self.IoT_n
        self.action_dim = self.action_S_dim + self.action_U_dim
        
        self.S_action_n =  2 * (self.Task_type_n + 1) * self.UAV_n
        self.U_action_n = 3*self.UAV_n + 2*self.IoT_n
        self.state_n = 1 + self.Task_type_n + self.Task_type_n * 2 + self.UAV_n * 4 + 8 * self.IoT_n
        self.action_n = self.S_action_n + self.U_action_n
        
        self.cost0 = -10

    def reset(self):
        self.Task_all_n = Task_all[np.random.randint(0, Task_all.shape[0])]
        self.IoT_Task_type = np.random.randint(0, self.Task_type_n, (self.IoT_n,),dtype=np.int32)
        self.T_n = 0
        self.UAV_E = np.ones((self.UAV_n,)) * self.args.E_UAV_max
        self.slice_comp = np.zeros((self.UAV_n, (self.Task_type_n + 1)))
        self.slice_comm = np.zeros((self.UAV_n, (self.Task_type_n + 1)))
        self.slice_act = -1
        self.UAV_IoT = np.zeros((self.IoT_n,),dtype=np.int32)
        self.UAV_xyz = self.UAV_xyz_init[:self.UAV_n, :]
        return self.get_state()

    def gen_Task(self):
        # Generate computational tasks based on the distribution (mean, variance, etc.) of data size, complexity, latency constraints, and task types of real computational tasks.
        self.Task = self.Task_all_n[self.T_n, :, :3]
        self.IoT_Task_type = self.Task_all_n[self.T_n, :, 3].astype(np.int32)

    def tmp_sl_action(self, slice_act):
        slice_act = (slice_act + 1 + 1e-20) / 2
        new_slice_comp = slice_act[0 : (self.Task_type_n + 1)*self.UAV_n].reshape((self.UAV_n, (self.Task_type_n + 1)))
        new_slice_comp = new_slice_comp / new_slice_comp.sum(axis=1, keepdims=True)

        new_slice_comm = slice_act[(self.Task_type_n + 1)*self.UAV_n : 2*(self.Task_type_n + 1)*self.UAV_n].reshape((self.UAV_n, (self.Task_type_n + 1)))
        new_slice_comm = new_slice_comm / new_slice_comm.sum()
        return new_slice_comp, new_slice_comm

    def new_state_cost(self, slice_act):
        slice_act = (slice_act[0] + 1 + 1e-20) / 2
        new_slice_comp = slice_act[0 : (self.Task_type_n + 1)*self.UAV_n].reshape((self.UAV_n, (self.Task_type_n + 1)))
        new_slice_comp = new_slice_comp / new_slice_comp.sum(axis=1, keepdims=True)

        new_slice_comm = slice_act[(self.Task_type_n + 1)*self.UAV_n : 2*(self.Task_type_n + 1)*self.UAV_n].reshape((self.UAV_n, (self.Task_type_n + 1)))
        new_slice_comm = new_slice_comm / new_slice_comm.sum()

        state = []
        state.append(self.T_n / self.args.T_max)
        for n in range(self.UAV_n):
            for i in range(self.Task_type_n):
                state.append((self.IoT_Task_type==i).sum() / 10)
                state.append(new_slice_comp[n,i])
                state.append(new_slice_comm[n,i])

        for n in range(self.UAV_n):
            state.extend(list((self.UAV_xyz[n, :] - np.array([self.args.x_min, self.args.y_min, self.args.z_min])) / (np.array([self.args.x_max, self.args.y_max, self.args.z_max]) - np.array([self.args.x_min, self.args.y_min, self.args.z_min]))))
            state.append(self.UAV_E[n] / self.args.E_UAV_max)
        
        for m in range(self.IoT_n): 
            state.extend(list((self.IoT_xy[m, :] - np.array([self.args.x_min, self.args.y_min])) / (np.array([self.args.x_max, self.args.y_max]) - np.array([self.args.x_min, self.args.y_min]))))
            state.extend([1 if i == self.IoT_Task_type[m] else 0 for i in range(self.Task_type_n)])
            state.extend((self.Task[m, :] - np.array([Task_size_min, Task_cpu_min, Task_delay_min])) / (np.array([Task_size_max, Task_cpu_max, Task_delay_max]) - np.array([Task_size_min, Task_cpu_min, Task_delay_min])))
        
        slice_comp_reCost = new_slice_comp[:, :-1] - self.slice_comp[:, :-1]
        slice_comp_reCost = (slice_comp_reCost > 0) * slice_comp_reCost
        slice_comm_reCost = new_slice_comm[:, :-1]- self.slice_comm[:, :-1]
        slice_comm_reCost = (slice_comm_reCost > 0) * slice_comm_reCost
        cost = (self.cost0 - (slice_comm_reCost.sum()+ slice_comp_reCost.sum()) * 10)
        return np.array(state), cost
    def cost_fun(self,slice_comm_reCost, slice_comp_reCost):
        return self.cost0 - (slice_comm_reCost.sum()+ slice_comp_reCost.sum()) * 10
    
    def apply_slice_action(self, new_slice_comp, new_slice_comm):
        self.slice_comp_reCost = new_slice_comp[:, :-1] - self.slice_comp[:, :-1]
        self.slice_comp_reCost = (self.slice_comp_reCost > 0) * self.slice_comp_reCost
        self.slice_comp = new_slice_comp
        self.slice_comm_reCost = new_slice_comm[:, :-1]- self.slice_comm[:, :-1]
        self.slice_comm_reCost = (self.slice_comm_reCost > 0) * self.slice_comm_reCost
        self.slice_comm = new_slice_comm

    def apply_user_action(self, user_act):
        user_act = (user_act + 1 + 1e-20) / 2
        self.IoT_comp = np.zeros((self.IoT_n,))
        self.IoT_comm = np.zeros((self.IoT_n,))
        self.UAV_dt_xyz = (user_act[: 3 * self.UAV_n] * 2 - 1).reshape((self.UAV_n, 3)) * np.array([self.args.v_uav_x_max, self.args.v_uav_y_max, self.args.v_uav_z_max] * self.UAV_n).reshape((self.UAV_n, 3))
        for n in range(self.UAV_n):
            for i in range(self.Task_type_n):
                tmp_comp = user_act[3*self.UAV_n: 3*self.UAV_n + self.IoT_n] * (self.IoT_Task_type == i)
                tmp_comp = tmp_comp / (tmp_comp.sum() + 1e-20)
                tmp_comm = user_act[3*self.UAV_n + self.IoT_n : 3*self.UAV_n + self.IoT_n * 2] * (self.IoT_Task_type == i)
                tmp_comm = tmp_comm / (tmp_comm.sum() + 1e-20)
                for m in range(self.IoT_n):
                    if self.IoT_Task_type[m] == i:
                        self.IoT_comp[m] = tmp_comp[m] * self.slice_comp[n][i]
                        self.IoT_comm[m] = tmp_comm[m] * self.slice_comm[n][i]
    def get_state(self):
        self.gen_Task()
        self.IoT_xy = self.Task_all_n[self.T_n, :, 4:]
        state = []
        state.append(self.T_n / self.args.T_max)
        for n in range(self.UAV_n):
            for i in range(self.Task_type_n):
                state.append((self.IoT_Task_type==i).sum() / 10)
                state.append(self.slice_comp[n,i])
                state.append(self.slice_comm[n,i])
        for n in range(self.UAV_n):
            state.extend(list((self.UAV_xyz[n, :] - np.array([self.args.x_min, self.args.y_min, self.args.z_min])) / (np.array([self.args.x_max, self.args.y_max, self.args.z_max]) - np.array([self.args.x_min, self.args.y_min, self.args.z_min]))))
            state.append(self.UAV_E[n] / self.args.E_UAV_max)
        for m in range(self.IoT_n):
            state.extend(list((self.IoT_xy[m, :] - np.array([self.args.x_min, self.args.y_min])) / (np.array([self.args.x_max, self.args.y_max]) - np.array([self.args.x_min, self.args.y_min]))))
            state.extend([1 if i == self.IoT_Task_type[m] else 0 for i in range(self.Task_type_n)])
            state.extend((self.Task[m, :] - np.array([Task_size_min, Task_cpu_min, Task_delay_min])) / (np.array([Task_size_max, Task_cpu_max, Task_delay_max]) - np.array([Task_size_min, Task_cpu_min, Task_delay_min])))
        return np.array(state)

    def step(self, act):
        slice_act, user_act  = act[:self.S_action_n], act[self.S_action_n:]
        new_slice_comp, new_slice_comm = self.tmp_sl_action(slice_act)
        if np.abs(new_slice_comp - self.slice_comp).max() > 1e-6 or np.abs(new_slice_comm - self.slice_comm).max() > 1e-6:
            self.apply_slice_action(new_slice_comp, new_slice_comm)
            self.slice_act = slice_act
            if self.T_n == 0:
                self.cost = self.cost0 
            else:
                self.cost = self.cost_fun(self.slice_comm_reCost, self.slice_comp_reCost)
        else:
            self.cost = 0
        self.apply_user_action(user_act)

        Power_I2U = np.zeros((self.IoT_n, self.UAV_n))
        Dist_I2U_3d = np.ones((self.IoT_n, self.UAV_n))
        for m in range(self.IoT_n):
            for n in range(self.UAV_n):
                dist_3_I2U = (self.UAV_xyz[n][0] - self.IoT_xy[m][0]) ** 2 + (self.UAV_xyz[n][1] - self.IoT_xy[m][1]) ** 2 + self.UAV_xyz[n][2] ** 2
                Dist_I2U_3d[m][n] = dist_3_I2U 
                h_C = (5 * 15 * 3.5 * 3.5) / (4 * 4 * np.pi * np.pi * dist_3_I2U * dist_3_I2U)
                Power_I2U[m][n] = self.IoT_tra_power[m] * h_C
        
        Need_Comp_list = []
        T_tra_list = np.zeros(self.IoT_n)
        T_local_list = np.zeros(self.IoT_n)
        for m in range(self.IoT_n):  
            if self.UAV_IoT[m] != -1:  
                intended_power = Power_I2U[m][self.UAV_IoT[m]]  
                sigma2 = 1e-16
                SINR = intended_power / (sigma2 + 1e-20)
                R = self.IoT_comm[m] * self.args.W * np.log2(1 + abs(SINR))  
                T_tra = (self.Task[m,0]) / (R + 1e-20)
                T_tra_list[m] = T_tra
                T_local_list[m] = 0
                if T_tra < self.Task[m,2]:
                    tmp1 = ((self.Task[m,0] * self.Task[m,1]) / (self.Task[m,2] - T_tra)) /  self.args.CPU_F_U
                    Need_Comp_list.append(tmp1)
                
        Need_Comp_list = np.array(Need_Comp_list)
        tmp_total_comp = self.slice_comp.copy()
        for m in np.argsort(Need_Comp_list):
            if tmp_total_comp[int(self.UAV_IoT[m]), int(self.IoT_Task_type[m])] > Need_Comp_list[m]:
                tmp_total_comp[int(self.UAV_IoT[m]), int(self.IoT_Task_type[m])] -= Need_Comp_list[m]
                self.IoT_comp[m] = Need_Comp_list[m]
            else:
                if tmp_total_comp[int(self.UAV_IoT[m]), int(self.IoT_Task_type[m])] > 0:
                    self.IoT_comp[m] = tmp_total_comp[int(self.UAV_IoT[m]), int(self.IoT_Task_type[m])] 
                    tmp_total_comp[int(self.UAV_IoT[m]), int(self.IoT_Task_type[m])] = -0.00001
                else:
                    self.IoT_comp[m] = 0

        is_success_task = np.zeros(self.IoT_n)
        delay_gap = np.zeros(self.IoT_n)
        for m in range(self.IoT_n):
            T_UAV_comput = (self.Task[m,0] * self.Task[m,1]) / (self.args.CPU_F_U * self.IoT_comp[m] + 1e-20)
            delay = min(T_tra_list[m] + T_UAV_comput, Task_delay_max)
            if delay < self.Task[m,2] + 0.00001:
                delay_gap[m] = 1 - delay / self.Task[m,2]
                is_success_task[m] = 1
        
        reward = 0
        for m in range(self.IoT_n):  
            if is_success_task[m] == 1: 
                pass
            else:
                reward = reward - 2

        self.UAV_xyz = self.UAV_xyz + self.UAV_dt_xyz
        for n in range(self.UAV_n): 
            if (self.UAV_xyz[n, :] >= np.array([self.args.x_min, self.args.y_min, self.args.z_min])).all() and (
                        self.UAV_xyz[n, :] <= np.array([self.args.x_max, self.args.y_max, self.args.z_max])).all():
                pass
            else:  
                self.UAV_xyz[n, :] = np.clip(self.UAV_xyz[n, :], np.array([self.args.x_min, self.args.y_min, self.args.z_min]), np.array([self.args.x_max, self.args.y_max, self.args.z_max]))

            UAV_move_dist = np.sqrt(self.UAV_dt_xyz[n][0] ** 2. + self.UAV_dt_xyz[n][1] ** 2 + self.UAV_dt_xyz[n][2] ** 2)
            EU_fly = (term1 + term2 + term3) * (UAV_move_dist / V_m) + (P_bla + P_ind) * T_n_hover
            
            # The communication energy consumption of UAVs is far less than the flight energy consumption, so we ignore the communication energy consumption of drones like most existing work.
            self.UAV_E[n] -= EU_fly 
            reward -=  1e-10*EU_fly
            if self.UAV_E[n] < 0:  
                self.UAV_E[n] = 0

        

        rr1 = 1 - self.slice_comp[:,-1].mean()
        rr2 = 1 - self.slice_comm[:,-1].sum()
        reward = reward - (rr1+rr2) * 2
        reward = reward + self.cost

        self.T_n += 1
        if self.T_n >= self.args.T_max:
            done = True
        else:
            done = False
        
        info = {}
        # reward shaping
        reward = reward
        return self.get_state(), reward, done, info
