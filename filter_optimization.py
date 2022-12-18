import numpy as np
# import visualize_characteristic
import matplotlib.pyplot as plt

def visualize_characteristic(H, omega_list, save_file_name):
    h = np.zeros(len(omega_list), dtype= np.complex)
    for i, omg in enumerate(omega_list):
        h[i] = H(omg)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    amp_h = np.abs(h) #20 * np.log10(np.abs(h) +  1e-20)
    angle_h = np.unwrap(np.angle(h))
    ax1.plot(omega_list, amp_h, marker="o")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(omega_list, angle_h, marker="o")

    fig.savefig(save_file_name)
    plt.close()

def compute_mean_square_error(H_, ref_h, omega_list):

    h = np.zeros(len(omega_list), dtype= np.complex)
    for i, omg in enumerate(omega_list):
        h[i] = H_(omg)

    e = np.real((ref_h - h) * np.conjugate(ref_h - h))
    return np.mean(e) 


def compute_abs_max_error(H_, ref_h, omega_list):

    h = np.zeros(len(omega_list), dtype= np.complex)
    for i, omg in enumerate(omega_list):
        h[i] = H_(omg)
    # e = np.real((ref_h - h) * np.conjugate(ref_h - h))

    e = np.abs(ref_h - h)
    return np.max(e)



def transfer_func(a0, pole_array, zero_array, omega):
    denomi = 1.0
    for p in  pole_array:
        denomi *= (1 - p * np.exp(-1j * omega)) * (1 - np.conjugate(p) * np.exp(-1j * omega))

    numer = 1.0
    for z in zero_array:
        numer *= (1 - z * np.exp(-1j * omega)) * (1 - np.conjugate(z) * np.exp(-1j * omega))
    
    return a0 * (numer/denomi)


def obj_func(pole_array, zero_array, a0, ref_h, omega_list, cs, is_mse = True):
    H_ = lambda omega : transfer_func(a0, pole_array, zero_array, omega)
    max_p2 = np.max(np.real(pole_array * np.conjugate(pole_array)))
    if is_mse:
        e = compute_mean_square_error(H_, ref_h, omega_list)
    else:
        e = compute_abs_max_error(H_, ref_h, omega_list)

    return e + (cs * max_p2 if max_p2 >= 1.0 else 0.0) 



def feedback_filter(x, pole):
    b1 = -np.real(pole) * 2
    b2 = np.real(pole * np.conjugate(pole))
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        s = x[i]
        if(i - 1 >= 0):
            s += -b1 * y[i - 1]
        if(i - 2 >= 0):
            s += -b2 * y[i - 2]
        y[i] = s
    return np.copy(y)

def feedforward_filter(x, zero):
    a1 = -np.real(zero) * 2
    a2 = np.real(zero * np.conjugate(zero))
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        s = x[i]
        if(i - 1 >= 0):
            s += a1 * x[i - 1]
        if(i - 2 >= 0):
            s += a2 * x[i - 2]
        y[i] = s
    return np.copy(y)

def filtering(x, pole_array, zero_array, a0):
    y = np.copy(x)
    for z in zero_array:
        y = feedforward_filter(y, z)
    for p in pole_array:
        y = feedback_filter(y, p)

    y *= a0
    return y



def annealing(HD, omega_list, M, N, cs, T, alpha, is_mse = True,
    init_a0 = None, init_pole_array = None, init_zero_array = None):
    h = np.zeros(len(omega_list), dtype=np.complex)
    for i, omg in enumerate(omega_list):
        h[i] = HD(omg)

    if init_a0 is None:
        a0 = 1.0
    else:
        a0 = init_a0
    if init_pole_array is None:
        pole_array = np.random.randn(M//2) + 1j * np.random.randn(M//2) 
    else:
        pole_array = np.copy(init_pole_array)
    
    if init_zero_array is None:
        zero_array = np.random.randn(N//2) + 1j * np.random.randn(N//2)
    else:
        zero_array = np.copy(init_zero_array)


    best_pole_array = np.copy(pole_array)
    best_zero_array = np.copy(zero_array)
    best_a0 = a0

    F = lambda pole_array_, zero_array_, a0_: \
        obj_func(pole_array_, zero_array_, a0_, h, omega_list, cs, is_mse)

    now_cost = F(best_pole_array, best_zero_array, best_a0)
    best_cost = now_cost


    cnt = 0
    while(True):
        if(cnt > 10000):
            break
        k = np.random.randint(0, M//2 + N//2 + 1)

        save_a0 = a0
        save_pole_array = np.copy(pole_array)
        save_zero_array = np.copy(zero_array)

        a0 += np.random.randn() * 0.01
        pole_array += (np.random.randn(M//2) + 1j * np.random.randn(M//2)) * 0.01
        zero_array += (np.random.randn(N//2) + 1j * np.random.randn(N//2)) * 0.01

        
        tmp_cost = F(pole_array, zero_array, a0)
        if(np.exp((now_cost - tmp_cost)/T) > np.random.rand() ):
        # if(((now_cost - tmp_cost)) >  np.log(np.random.rand()) * T):
        # if(tmp_cost < now_cost):
            now_cost = tmp_cost
            print("now_cost = ", now_cost, T, cnt)
            if(now_cost < best_cost):
                best_a0 = a0
                best_pole_array = np.copy(pole_array)
                best_zero_array = np.copy(zero_array)
                best_cost = now_cost
                print("best_cost = ", best_cost)
        else:
            a0 = save_a0
            pole_array = np.copy(save_pole_array)
            zero_array = np.copy(save_zero_array)

            # if(k <= 0):
            #     a0 = save
            # elif(k <= M//2):
            #     pole_array[k - 1] = save
            # else:
            #     zero_array[k - (M//2 + 1)] = save
        cnt += 1        
        T *= alpha
    return best_a0, best_pole_array, best_zero_array, best_cost


np.random.seed(0)
H = lambda omg :  np.exp(- 1j * 12 * omg) if omg >= 2 * 0.2 * np.pi and omg <= 2 * 0.3 * np.pi   else 0.0
# H = lambda omg : (1 + np.exp(-2j * omg))/(1 - 0.5 * np.exp(-1j * omg) + 0.3 * np.exp(-2j*omg) +  0.2 * np.exp(-3j*omg) -  0.1 * np.exp(-4j*omg))
# H = lambda omg : np.exp(- 1j * 2 * omg) if (omg >= 2 * 0.1 * np.pi and omg <= 2 * 0.15 * np.pi) or (omg >= 2 * 0.3 * np.pi and omg <= 2 * 0.4 * np.pi)    else 0.0
# omega_list1 = np.linspace(0, 2 * 0.07 * np.pi , 200)
# omega_list2 = np.linspace(2 * 0.1 * np.pi, 2 * 0.15 * np.pi , 200)
# omega_list3 = np.linspace(2 * 0.18 * np.pi, 2 * 0.27 * np.pi , 200)
# omega_list4 = np.linspace(2 * 0.3 * np.pi, 2 * 0.4 * np.pi , 200)
# omega_list5 = np.linspace(2 * 0.42 * np.pi,  np.pi , 200)
# omega_list = np.concatenate([omega_list1, omega_list2, omega_list3, omega_list4, omega_list5])


omega_list1 = np.linspace(0, 2 * 0.15 * np.pi , 200)
omega_list2 = np.linspace(2 * 0.2 * np.pi, 2 * 0.3 * np.pi , 200)
omega_list3 = np.linspace(2 * 0.35 * np.pi,  np.pi , 200)
omega_list = np.concatenate([omega_list1, omega_list2, omega_list3])

# omega_list_all = np.linspace(0, np.pi , 200)

T = 0.4
Fs = 300/1
t_list = np.arange(0, T, 1/Fs)

x = np.sin(2 * 0.02 * Fs/2 * np.pi * t_list) + \
    np.sin(2 * 0.4 *  Fs/2 * np.pi * t_list) + \
    np.sin(2 * 0.8 *  Fs/2 * np.pi * t_list)
# x =  np.cos(2 * 0.02 * Fs/2 * np.pi * t_list)
# x =  np.sin(2 * 0.4 *  Fs/2 * np.pi * t_list) 
# x =  np.cos(2 * 0.9 *  Fs/2 * np.pi * t_list)
plt.plot(t_list, x, marker="o")
plt.savefig("origin.png")

visualize_characteristic(H, omega_list, "ref.png")

best_cost = 1e9
for i in range(5):
    a0_, pole_array_, zero_array_, cost = annealing(H, omega_list, 8, 8, 50, 10, 0.998, is_mse = False)
    if(cost < best_cost):
        a0 = a0_
        pole_array = np.copy(pole_array_)
        zero_array = np.copy(zero_array_)
        best_cost = cost

# a0, pole_array, zero_array = annealing(H, omega_list, 8, 8, 5.0, 10, 0.998, a0, pole_array, zero_array)
# a0, pole_array, zero_array = annealing(H, omega_list, 8, 8, 5.0, 10, 0.998, a0, pole_array, zero_array)
# a0, pole_array, zero_array = annealing(H, omega_list, 8, 8, 5.0, 10, 0.998, a0, pole_array, zero_array)

# a0, pole_array, zero_array, _ = annealing(H, omega_list, 8, 8, 5.0, 1, 0.99, a0, pole_array, zero_array, False)


for p in pole_array:
    print(np.abs(p))


H_ = lambda omega : transfer_func(a0, pole_array, zero_array, omega)

h = np.zeros(len(omega_list), dtype=np.complex)
for i, omg in enumerate(omega_list):
    h[i] = H(omg)
print(compute_abs_max_error(H_, h, omega_list))
visualize_characteristic(H_, omega_list, "opt.png")



y = filtering(x, pole_array, zero_array, a0)

plt.plot(t_list, y)


plt.savefig("filtered.png")