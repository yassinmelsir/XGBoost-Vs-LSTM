from xg.xg import run_xg
from lstm.lstm import run_lstm
from ts.ts import run_ts_fs

F_history = run_ts_fs('xg', 5)
print(F_history)    