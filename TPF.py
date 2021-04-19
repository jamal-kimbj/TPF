from datetime import datetime
import numpy as np
import pandas as pd
import scipy.stats as stats
from dataturbo import DataTurbo
from dateutil.relativedelta import relativedelta
import quantrading as qt
import matplotlib.pyplot as plt
from time import time
from scipy.stats import lognorm
import os


class MakingPrice:
    def __init__(self, year, start_date):
        self.year = year
        self.start_date = start_date

        end_date = datetime.strptime(self.start_date, "%Y-%m-%d") + relativedelta(months=year * 12)
        end_date = datetime.strftime(end_date, "%Y-%m-%d")
        self.dt_index = pd.date_range(start=self.start_date, end=end_date)

    def GBMsimulator(self, seed, So, mu, sigma, Cov, T, N):
        """
        Parameters
        seed:   seed of simulation
        So:     initial stocks' price
        mu:     expected return
        sigma:  volatility
        Cov:    covariance matrix
        T:      time period
        N:      number of increments
        """
        np.random.seed(seed)
        dim = np.size(So)
        t = np.linspace(0., T, int(N))
        A = np.linalg.cholesky(Cov)
        S = np.zeros([dim, int(N)])
        S[:, 0] = So
        for i in range(1, int(N)):
            drift = (mu - 0.0 * sigma ** 2) * (t[i] - t[i - 1])
            Z = np.random.normal(0., 1., dim)
            diffusion = np.matmul(A, Z) * (np.sqrt(t[i] - t[i - 1]))
            S[:, i] = S[:, i - 1] * np.exp(drift + diffusion)
        return S, t


    def make_simulation_price(self, mu, sigma, Cov):
        seed = int(np.random.uniform(1, 2 ** 32 - 1, 1))

        T = self.year
        N = int(len(self.dt_index))
        S0 = np.array([100 for x in range(0, len(mu))])
        discount = 1#0.96
        stocks, time = self.GBMsimulator(seed, S0, mu, sigma, Cov * (discount ** 2), T, N)
        stocks = pd.DataFrame(stocks).T
        stocks.index = self.dt_index

        return stocks

    def simulation(self, simul, mu_list, sigma_list, corr):
        mu = np.array(mu_list) * (12/Delta_time)
        sigma = np.array(sigma_list) * np.sqrt(12/Delta_time)

        if type(corr) != pd.DataFrame:
            Cov = np.tensordot(sigma, sigma, 0) * np.array(np.eye(len(mu_list)))
            corr = pd.DataFrame(np.eye(len(mu_list)))
        else:
            Cov = np.tensordot(sigma, sigma, 0) * np.array(corr)
            corr = pd.DataFrame(corr)

        simul_list = []
        simulation_price_df = self.make_simulation_price(mu, sigma, Cov)
        simul_list.append(simulation_price_df)
        for i in range(1, simul):
            simulation_price_df = self.make_simulation_price(mu, sigma, Cov)
            simul_list.append(simulation_price_df)
        return simul_list

    def simulation_check(self, result, mu_list, sigma_list, corr):
        mu = np.array(mu_list) * (12/Delta_time)
        sigma = np.array(sigma_list) * np.sqrt(12/Delta_time)
        pct_change = np.log(result[0]).diff().dropna().describe()
        mu_check_list = (pct_change.loc['mean']) * 365
        sigma_check_list = (pct_change.loc['std']) * np.sqrt(365)
        corr_check_list = (result[0].pct_change().corr())
        for i in range(1, simul):
            pct_change = np.log(result[i]).diff().dropna().describe()
            mu_check_list += (pct_change.loc['mean']) * 365
            sigma_check_list += (pct_change.loc['std']) * np.sqrt(365)
            corr_check_list += (result[i].pct_change().corr())

        return (mu / (mu_check_list / simul)), \
               (sigma / (sigma_check_list / simul)), \
               pd.DataFrame(np.array(corr) / np.array((corr_check_list / simul)))



def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


##현재상태에서 다음 상태까지의 표준화(수익률) 거리를 계산합니다.
def make_standard_distance_list(mu_list, sigma_list, Wj_list, Wi, cash_flow=0):
    standard_distance_list = []
    for i in range(0, len(mu_list)):
        mu = mu_list[i]
        sigma = sigma_list[i]
        standard = (np.log((Wj_list -0.5) / (Wi + cash_flow)) - (mu)) / sigma
        standard_distance_list.append(standard)
    standard_distance_list = np.reshape(standard_distance_list,
                                        (len(standard_distance_list), len(standard_distance_list[0])))
    return standard_distance_list


##표준화(수익률) 거리로 갈 수 있는 확률을 계산합니다.
# TODO PDF 확률 구하는법 점추정 방법으로 수정 필요
def make_prob_list(standard_distance_list):
    cdf = 1 - stats.norm(0, 1).cdf(standard_distance_list)
    pdf = cdf[:, 1:] - cdf[:, :-1]
    pdf = np.concatenate((np.reshape(cdf[:, 0], (len(cdf), 1)), pdf), axis=1)
    pdf[:, -1] = 1 - np.sum(pdf[:, :-1], 1)
    return pdf

def make_map(market_signal, directory, file_name):
    ### 폴더 생성 ###
    createFolder(directory + '\\' +file_name)
    # 금액 그리드를 생성한다.
    account_idx = np.array([1 + x for x in range(0, N)], dtype=float)
    # 유틸리티 함수 현재는 디지털(목표 달성)
    final_value = np.array([1 if x >= (RP) else 0 for x in account_idx])
    if maturity_penalty == 'mp':
        # 만기시점 페널티 값 -1
        final_value_mp = np.array([-1 if x <= (total_account * maturity_const) else 0 for x in account_idx])
        final_value = final_value + final_value_mp
    # map 공간 데이터 플레임 만들기
    total_value_df = pd.DataFrame(index=account_idx, columns=[x for x in range(int((year * 12) / Delta_time + 1))])
    # 만기 보상값을 map에 설정
    total_value_df.iloc[:, -1] = final_value
    # 1000부터 시작하도록 뒤집는다. 이유는 모르겠다...
    total_value_df = total_value_df[::-1]
    total_action_df = total_value_df.copy()
    # Wj_list 는 한스텝 뒤의 공간을 의미
    # Wi_list 는 현재시점의 공간을 의미
    Wj_list = np.array(total_value_df.index)
    Wi_list = np.array(total_value_df.index)
    # 적립금을 반영하기 위해 사용되는 변수
    check_count = total_account
    # n 은 시점 (만기부터 시작)
    for n in total_value_df.columns[:-1][::-1]:
        if path_penalty != 'pp':
            Wj_value = np.array(total_value_df.iloc[:, n + 1])
        else:
            temp = total_value_df.iloc[:, n + 1]
            # 지금까지 쌓여있는 원금 * path_const 이하인 부분 -1로 패널라이즈
            idx = total_value_df.index[total_value_df.index < check_count * path_const]
            temp.loc[idx] = -1
            Wj_value = np.array(temp)
        Wi_value = np.array([0 for x in range(len(Wi_list))], dtype=float)
        Wi_action = np.array([0 for x in range(len(Wi_list))], dtype=float)
        # 마켓 시그널에 따른 분포를 다르게 가져간다.
        mu_list = total_mu_list[market_signal[n]]
        sigma_list = total_sigma_list[market_signal[n]]
        # 현재 가능한 모든 공간(금액)을 돌면서 밸류와 액션을 계산한다.
        for i in range(len(Wi_list)):
            Wi = Wi_list[i]
            standard_distance_list = make_standard_distance_list(mu_list, sigma_list, Wj_list, Wi, cash_flow)
            pdf = make_prob_list(standard_distance_list)
            mtx = Wj_value @ pdf.T
            Wi_value[i] = np.max(mtx)
            Wi_action[i] = np.argmax(mtx)
        ####밸류값 스무딩### n = 만기-1 부터 스무딩 적용됨
        value_array = Wi_value
        #value_array = np.append(value_array[0], (value_array[:-1] + value_array[1:]) / 2)
        total_value_df.iloc[:, n] = value_array
        total_action_df.iloc[:, n] = Wi_action
        # 적립식인 경우 적립금액을 빼준다. 거치식인경우 - 0
        check_count -= cash_flow
        print(check_count)
    total_action_df.to_csv(f'{directory}\\{file_name}_action.csv')
    total_value_df.to_csv(f'{directory}\\{file_name}_value.csv')

def make_state_prob_map(market_signal, action_df, directory, file_name):
    # 사전에 설정한 action_df 를 이용해서 디지털 보상으로 value 값을 구하면 그게 action 이 가진 확률이 됨
    account_idx = np.array([1 + x for x in range(0, N)], dtype=float)
    final_value = np.array([1 if x >= (RP) else 0 for x in account_idx])
    total_value_df = pd.DataFrame(index=account_idx, columns=[x for x in range(int((year * 12) / Delta_time + 1))])
    total_value_df.iloc[:, -1] = final_value
    total_value_df = total_value_df[::-1]

    Wj_list = np.array(total_value_df.index)
    Wi_list = np.array(total_value_df.index)

    check_count = total_account
    for n in total_value_df.columns[:-1][::-1]:
        Wj_value = np.array(total_value_df.iloc[:, n + 1])
        Wi_value = np.array([0 for x in range(len(Wi_list))], dtype=float)
        mu_list = total_mu_list[market_signal[n]]
        sigma_list = total_sigma_list[market_signal[n]]
        for i in range(len(Wi_list)):
            Wi = Wi_list[i]
            action = int(action_df[str(n)].loc[Wi])
            standard_distance_list = make_standard_distance_list(mu_list, sigma_list, Wj_list, Wi, cash_flow)
            pdf = make_prob_list(standard_distance_list)
            mtx = Wj_value @ pdf.T
            Wi_value[i] = mtx[action]

        prob_array = Wi_value
        #### 시간이 지남에 따라 확률 떨어지는 효과 보정###
        #prob_array = np.append(prob_array[0], (prob_array[:-1] + prob_array[1:]) / 2)
        total_value_df.iloc[:, n] = prob_array

        check_count -= cash_flow
        print(check_count)

    total_value_df.to_csv(f'{directory}\\{file_name}_state_prob.csv')


def make_dist_map(market_signal, action_df, directory, file_name):
    account_idx = np.array([1 + x for x in range(0, N)], dtype=float)
    total_value_df = pd.DataFrame(index=account_idx, columns=[x for x in range(int((year * 12) / Delta_time + 1))])
    total_value_df.iloc[: , :] = 0
    total_value_df = total_value_df[::-1]

    Wj_list = np.array(total_value_df.index)
    Wi_list = np.array(total_value_df.index)

    check_count = total_account

    Wi = initial_account
    action = int(action_df[str(0)].loc[Wi])
    standard_distance_list = make_standard_distance_list(total_mu_list[market_signal[0]], total_sigma_list[market_signal[0]], Wj_list, Wi, cash_flow)
    pdf = make_prob_list(standard_distance_list)
    total_value_df.iloc[:, 1] += pdf[action]
    # 분포를 구할때는 forward calculation 을 해야하므로 n은 1부터 돈다.
    for n in total_value_df.columns[1:-1]:
        Wi_value = np.array([0 for x in range(len(Wi_list))], dtype=float)
        mu_list = total_mu_list[market_signal[n]]
        sigma_list = total_sigma_list[market_signal[n]]
        for i in range(len(Wi_list)):
            Wi = Wi_list[i]
            action = int(action_df[str(n)].loc[Wi])
            ori_prob = total_value_df[n].loc[Wi]
            standard_distance_list = make_standard_distance_list(mu_list, sigma_list, Wj_list, Wi, cash_flow)
            pdf = make_prob_list(standard_distance_list)
            Wi_value += ori_prob * pdf[action]

        prob_array = Wi_value
        #### 확률보정 ####
        #prob_array = np.append((prob_array[:-1] + prob_array[1:]) / 2, prob_array[-1])

        total_value_df.iloc[:, n + 1] = prob_array

        check_count -= cash_flow
        print(check_count)

    total_value_df.to_csv(f'{directory}\\{file_name}_dist_map.csv')



def make_backtest_simulation(market_signal, test_date, index_df_list, action_df, prob_df, simulation_num):
    ### 시작일 ###
    index_df = index_df_list[0]
    start_date = datetime.strptime(test_date, "%Y-%m-%d")
    idx = index_df.index[index_df.index >= start_date]
    start_date = idx[np.argmin(np.abs(idx - pd.to_datetime(start_date)))]
    ### 종료일 ###
    final_date = start_date + relativedelta(months=maturity * Delta_time + 1)
    final_date = idx[np.argmin(np.abs(idx - pd.to_datetime(final_date)))]
    ### 시작일 ~ 종료일 ###
    sample_date = index_df.index[(index_df.index >= start_date) & (index_df.index <= final_date)]
    ### 비중을 받는 df (계좌와 포트가 일치해야함)###
    sample_weight_df = pd.DataFrame(index=sample_date, columns=index_df.columns)
    ### 계좌 수량을 받는 df ###
    sample_account_quantity_df = sample_weight_df.copy()
    ### bm포트 수량을 받는 df ###
    sample_bm_quantity_df = sample_weight_df.copy()
    ### 계좌 달성률을 받는 df ###
    sample_prob_df = pd.DataFrame(index=sample_date, columns=['prob'])
    ### 월초 리벨런싱을 위한 날짜 리스트 계산
    date_list = [sample_date[0]]
    for time in range(1, len(sample_date)):
        if sample_date[time].month != date_list[-1].month and len(date_list) < maturity + 1:
            date_list.append(sample_date[time])

    pct_list = []
    for i in range(len(market_signal)):
        #print(market_signal[i])
        market_regime_price_df = index_df_list[market_signal[i]]
        if i == 0:
            pct_list.append(market_regime_price_df.loc[(market_regime_price_df.index <= date_list[i + 1]) &
                                                (market_regime_price_df.index >= date_list[i])].pct_change().fillna(0))
        else:
            pct_list.append(market_regime_price_df.loc[(market_regime_price_df.index <= date_list[i + 1]) &
                                                (market_regime_price_df.index >= date_list[i])].pct_change().fillna(0).iloc[1:, :])
    pct_list.append(market_regime_price_df.loc[(market_regime_price_df.index <= final_date) &
                                        (market_regime_price_df.index >= date_list[i + 1])].pct_change().fillna(0).iloc[1:, :])

    total_regime_price_df = (pd.concat(pct_list, 0) + 1).cumprod()

    if len(sample_date) == len(total_regime_price_df):
        sample_price_df = market_regime_price_df
        ### 계좌상태 기록
        account_list = [initial_account + cash_flow]
        ### bm포트상태 기록
        bm_list = [initial_account]
        for i in range(0, len(date_list) - 1):
            old_date = date_list[i]
            new_date = date_list[i + 1]
            account_notional = round(account_list[i])
            bm_notional = round(bm_list[i])
            sample_weight_df.loc[old_date] = sample_weight_df.loc[old_date].fillna(0)
            action = action_df[str(i)].loc[account_notional]
            ## 시뮬레이션의 경우 액션별로 포트를 선행적으로 구한 후 체결만 시킴
            sample_weight_df.loc[old_date][action] = 1
            weight = sample_weight_df.loc[old_date]
            ## 액션에 해당하는 포트폴리오 수량 (계좌)
            account_quantity = (account_notional * weight / sample_price_df.loc[old_date][action])
            ## 액션에 해당하는 포트폴리오 수량 (기준가)
            bm_quantity = (bm_notional * weight / sample_price_df.loc[old_date][action])
            ### 현재계좌에 해당하는 달성확률 기록
            sample_prob_df.loc[old_date] = prob_df[str(i)].loc[account_notional]
            ### 현재계좌에 할당된 기초자산 수량
            sample_account_quantity_df.loc[old_date] = account_quantity
            ### bm포트에 할당된 기초자산 수량
            sample_bm_quantity_df.loc[old_date] = bm_quantity
            ### 계좌 수익 + 적립금기록
            account_profit = ((sample_price_df.loc[new_date] - sample_price_df.loc[old_date]) * account_quantity).sum()
            account_list.append(account_profit + account_list[-1] + cash_flow)
            ### bm포트 수익 기록
            bm_profit = ((sample_price_df.loc[new_date] - sample_price_df.loc[old_date]) * bm_quantity).sum()
            bm_list.append(bm_profit + bm_list[-1])
        ### 마지막에 추가로 들어온 적립금 제거
        account_list[-1] = account_list[-1] - cash_flow
        sample_account_quantity_df = sample_account_quantity_df.fillna(method='ffill')
        sample_bm_quantity_df = sample_bm_quantity_df.fillna(method='ffill')
        ### 계좌의 daily 상태 만듬 (가격 * 수량)
        sample_account = pd.DataFrame(np.diag(sample_price_df @ sample_account_quantity_df.T))
        ### bm의 daily 상태 만듬 (가격 * 수량)
        sample_bm = pd.DataFrame(np.diag(sample_price_df @ sample_bm_quantity_df.T))
        sample_account.index = sample_date
        sample_bm.index = sample_date
        print(account_list[-1])
        print(sample_account.iloc[-1])
        result = pd.concat([sample_bm, sample_account, sample_prob_df], 1)
        result.columns = ['BM', 'Account', 'Prob']
        result = pd.concat([result, sample_weight_df], 1).fillna(method='ffill')
        plt.plot(result['BM'])
        plt.show()

        result_path = os.getcwd() + path_name + '\\' + file_name + '\\simulation'
        createFolder(result_path)
        result.to_csv(f'{result_path}\\simulation{simulation_num}.csv', encoding='utf-8-sig')
        return result
    else:
        print('Error : 가격데이터 날짜가 일치하지 않습니다.')



def make_backtest(market_signal, test_date, index_df, action_df, prob_df):
    ### 시작일 ###
    start_date = datetime.strptime(test_date, "%Y-%m-%d")
    idx = index_df.index[index_df.index >= start_date]
    start_date = idx[np.argmin(np.abs(idx - pd.to_datetime(start_date)))]
    ### 종료일 ###
    final_date = start_date + relativedelta(months=maturity * Delta_time + 1)
    final_date = idx[np.argmin(np.abs(idx - pd.to_datetime(final_date)))]
    ### 시작일 ~ 종료일 ###
    sample_date = index_df.index[(index_df.index >= start_date) & (index_df.index <= final_date)]
    ### 기간에 일치하는 가격샘플 ###
    sample_price_df = index_df.loc[sample_date]
    sample_price_df = sample_price_df.fillna(method='ffill')
    ### 비중을 받는 df (계좌와 포트가 일치해야함)###
    sample_weight_df = pd.DataFrame(index=sample_price_df.index, columns=sample_price_df.columns)
    ### 계좌 수량을 받는 df ###
    sample_account_quantity_df = sample_weight_df.copy()
    ### bm포트 수량을 받는 df ###
    sample_bm_quantity_df = sample_weight_df.copy()
    ### 계좌 달성률을 받는 df ###
    sample_prob_df = pd.DataFrame(index=sample_price_df.index, columns=['prob'])
    ### 월초 리벨런싱을 위한 날짜 리스트 계산
    date_list = [sample_date[0]]
    for time in range(1, len(sample_price_df)):
        if sample_date[time].month != date_list[-1].month and len(date_list) < maturity + 1:
            date_list.append(sample_date[time])
    ### 계좌상태 기록
    account_list = [initial_account + cash_flow]
    ### bm포트상태 기록
    bm_list = [initial_account]
    for i in range(0, len(date_list) - 1):
        # TODO action weight list 는 2002년 전/후 설정이 안되어 있음 (전부 2002년 이후)
        action_weight = action_weight_list[market_signal[i]]
        old_date = date_list[i]
        new_date = date_list[i + 1]
        account_notional = round(account_list[i])
        bm_notional = round(bm_list[i])
        sample_weight_df.loc[old_date] = sample_weight_df.loc[old_date].fillna(0)
        action = action_df[str(i)].loc[account_notional]

        ## 실제가격의 경우 종가를 사용하여 체결시킴
        weight = action_weight.loc[action]
        sample_weight_df.loc[old_date] = weight
        ## 액션에 해당하는 포트폴리오 수량 (계좌)
        account_quantity = (account_notional * weight / sample_price_df.loc[old_date])
        ## 액션에 해당하는 포트폴리오 수량 (기준가)
        bm_quantity = (bm_notional * weight / sample_price_df.loc[old_date])
        ### 현재계좌에 해당하는 달성확률 기록
        sample_prob_df.loc[old_date] = prob_df[str(i)].loc[account_notional]
        ### 현재계좌에 할당된 기초자산 수량
        sample_account_quantity_df.loc[old_date] = account_quantity
        ### bm포트에 할당된 기초자산 수량
        sample_bm_quantity_df.loc[old_date] = bm_quantity
        ### 계좌 수익 + 적립금기록
        account_profit = ((sample_price_df.loc[new_date] - sample_price_df.loc[old_date]) * account_quantity).sum()
        account_list.append(account_profit + account_list[-1] + cash_flow)
        ### bm포트 수익 기록
        bm_profit = ((sample_price_df.loc[new_date] - sample_price_df.loc[old_date]) * bm_quantity).sum()
        bm_list.append(bm_profit + bm_list[-1])
    ### 마지막에 추가로 들어온 적립금 제거
    account_list[-1] = account_list[-1] - cash_flow
    sample_account_quantity_df = sample_account_quantity_df.fillna(method='ffill')
    sample_bm_quantity_df = sample_bm_quantity_df.fillna(method = 'ffill')
    ### 계좌의 daily 상태 만듬 (가격 * 수량)
    sample_account = pd.DataFrame(np.diag(sample_price_df @ sample_account_quantity_df.T))
    ### bm의 daily 상태 만듬 (가격 * 수량)
    sample_bm = pd.DataFrame(np.diag(sample_price_df @ sample_bm_quantity_df.T))
    sample_account.index = sample_date
    sample_bm.index = sample_date
    print(account_list[-1])
    print(sample_account.iloc[-1])
    result = pd.concat([sample_bm, sample_account, sample_prob_df], 1)
    result.columns = ['BM', 'Account', 'Prob']
    result = pd.concat([result, sample_weight_df], 1).fillna(method = 'ffill')
    plt.plot(result['BM'])
    plt.show()

    result_path = os.getcwd() + path_name + '\\' + file_name + '\\real'
    createFolder(result_path)
    result.to_csv(f'{result_path}\\{test_date}.csv', encoding = 'utf-8-sig')
    return result


################################################################

maturity_penalty = 'mp'  # 만기패널티 여부 'nomp'
path_penalty = 'pp'  # 패스패널티 여부 'pp'
year = 1
maturity_const = 0.9  # 만기패널티 값
path_const = 0.80  # 패스패널티 값
N = 1000  # 계좌의 최대 크기
Delta_time = 1  # 몇개월 리벨런싱이죠??
# TODO 금액 간격(수익률 간격 폭을 설정 가능하게 수정)
maturity = int((12 / Delta_time) * (year))
initial_account = 100  # 얼마로 시작하나요?? 디폴트 100
cash_flow = 0  # 적립금을 얼마씩 넣나요?? 없으면 0
# 12개월 이라면 총 11번의 적립 (마지막 시점제외)
total_account = (cash_flow * (year * (12 / Delta_time) -1) + initial_account)
RP = 1.07 * total_account  ##목표금액

### bear = 0, neutral = 1, bull = 2
#market_signal = [1 for x in range(0, maturity)]
###  market sigmal length should be equal to maturity
market_signal = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

### Action 에 대한 분포 파라메터 가정 (로그스케일로 가져가기)
total_mu_list = []
total_sigma_list = []
### bear case
total_mu_list.append(np.array([0.015, 0.028, 0.044, 0.055, 0.064, 0.072, 0.08, 0.085]) / (12 / Delta_time) * -1)
total_sigma_list.append(np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]) / np.sqrt(12 / Delta_time))
### neutral case
total_mu_list.append(np.array([0.015, 0.028, 0.044, 0.055, 0.064, 0.072, 0.08, 0.085]) / (12 / Delta_time))
total_sigma_list.append(np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]) / np.sqrt(12 / Delta_time))
### bull case
total_mu_list.append(np.array([0.015, 0.028, 0.044, 0.055, 0.064, 0.072, 0.08, 0.085]) / (12 / Delta_time))
total_sigma_list.append(np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]) / np.sqrt(12 / Delta_time))

#### 경로설정
project_name = 'simon'
path_name =f'\\{project_name}\\{year}year_{project_name}\\RP{round(RP * 100/total_account)}'
file_name =f'{maturity_penalty}{maturity_const}_{path_penalty}{path_const}_TPF'
directory = os.getcwd() + path_name


### 액션맵, 밸류맵, state별확률, dist맵 계산
make_map(market_signal, directory, file_name)
action_df = pd.read_csv(f'{directory}\\{file_name}_action.csv', index_col=0)
make_state_prob_map(market_signal, action_df, directory, file_name)
make_dist_map(market_signal, action_df, directory, file_name)


###############################################################################################################
## 실제 데이터를 이용한 Backtest code ############################################################################
###############################################################################################################

action_weight_list = []
action_weight_list.append(pd.read_excel('action_weight.xlsx', sheet_name='new_bear', index_col=0).iloc[:, 2:])
action_weight_list.append(pd.read_excel('action_weight.xlsx', sheet_name='new_neutral', index_col=0).iloc[:, 2:])
action_weight_list.append(pd.read_excel('action_weight.xlsx', sheet_name='new_bull', index_col=0).iloc[:, 2:])

index_df = pd.read_excel('price_df.xlsx', sheet_name='price_df_old', index_col=0).iloc[:,
           :len(action_weight_list[0].columns)]

action_df = pd.read_csv(f'{directory}\\{file_name}_action.csv', index_col=0)
prob_df = pd.read_csv(f'{directory}\\{file_name}_state_prob.csv', index_col=0)

# --------------------------------
# 특정 시작 날짜, 만기에 따른 Backtest
# --------------------------------
# TODO backtest 시에, Market_signal 미래참조 수정 필요
initial_date = '1998-01-01'
result_list = [make_backtest(market_signal, initial_date, index_df, action_df, prob_df)]

# ------------------------------------------------------------------------------------------------
# rolling Backtest / initial 날짜로부터 특정 개월 간격으로 특정회수의 Backtest 를 수행
# ------------------------------------------------------------------------------------------------
month_step = 6
backtest_nb = 30
for i in range(0, backtest_nb):
    print(i)
    ######### 몇 개월마다 찍을것이오?? #####
    new_date = datetime.strptime(initial_date, "%Y-%m-%d") + relativedelta(months=int(month_step))
    result_list.append(make_backtest(market_signal, new_date.strftime("%Y-%m-%d"), index_df, action_df, prob_df))
    initial_date = new_date

####################################################################################################
# monte carlo simulation by random price (generation method : cholesky decompostion)
# TODO (수정 필요) 개별 자산에 대한 가격생성 -> 포트밸류 생성
####################################################################################################

bear_price_df = (index_df.pct_change() @ action_weight_list[0].T + 1).cumprod().dropna().loc[:, ::-1]
neutral_price_df = (index_df.pct_change() @ action_weight_list[1].T + 1).cumprod().dropna().loc[:, ::-1]
bull_price_df = (index_df.pct_change() @ action_weight_list[2].T + 1).cumprod().dropna().loc[:, ::-1]


simul = 30
### Correlation 고려안할꺼면 마지막 값 0 ###
making = MakingPrice(year + 1, '2001-01-01')

bear_result = making.simulation(simul, total_mu_list[0], total_sigma_list[0], bear_price_df.corr())
check = making.simulation_check(bear_result, total_mu_list[0], total_sigma_list[0], bear_price_df.corr())

neutral_result = making.simulation(simul, total_mu_list[1], total_sigma_list[1], neutral_price_df.corr())
bull_result = making.simulation(simul, total_mu_list[2], total_sigma_list[2], bull_price_df.corr())


result_list = []
for i in range(simul):
    simul_price_list = [bear_result[i]]
    simul_price_list.append(neutral_result[i])
    simul_price_list.append(bull_result[i])
    result_list.append(make_backtest_simulation(market_signal, '2001-01-01', simul_price_list, action_df, prob_df, i + 1))