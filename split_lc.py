import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
#import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Light Curve Splitter")
    parser.add_argument("filename", type=str, help="Path to the input data file")
    parser.add_argument("--output", type=str, default="ranges.!", help="Path to the output file")
    parser.add_argument("--preview", type=str, default="SplitPreview.html", help="Path to the output preview file")
    parser.add_argument("--epoch", type=np.float64, help="Initial Epoch")
    parser.add_argument("--period", type=np.float64, help="Period")
    parser.add_argument("--start-phase", type=np.float64, help="Start phase")
    parser.add_argument("--stop-phase", type=np.float64, help="Stop phase")
    return parser.parse_args()

args = parse_args()
#print(vars(args))
#sys.exit()

data_file_name = args.filename
output_file_name = args.output
preview_file_name = args.preview
epoch = args.epoch
period = args.period
start_phase = args.start_phase
stop_phase = args.stop_phase

data = pd.read_csv(data_file_name, 
                   comment='#', skip_blank_lines=True,
                   sep="\\s+",
                   names=['time', 'mag'],
                   dtype={'time': 'float64', 'mag': 'float64'},
                   usecols=['time', 'mag'])

times = data['time']
mags  = data['mag']

# Sort by times (just in case)
times, mags = zip(*sorted(zip(times, mags)))
times = np.array(times)
mags = np.array(mags)

min_time = np.min(times)
max_time = np.max(times)

min_cycle = int(round((min_time-epoch)/period, 0))
max_cycle = int(round((max_time-epoch)/period, 0))

with open(preview_file_name, "w") as f_preview:
    f_preview.write("<html><body>\n")
    f_preview.write("<h2>Preview</h2>\n")
    f_preview.write("<hr>\n")

n = 0
with open(output_file_name, "w") as f:
    for i in range(min_cycle, max_cycle + 1):
        t_start = epoch + i * period + start_phase * period
        t_stop  = epoch + i * period + stop_phase  * period
        mask = (times >= t_start) & (times <= t_stop)
        idx = np.where(mask)[0]
        #f.write(f'{idx}\n')
        info_str = ''
        if len(idx) > 0:
            n += 1
            info_str = f'{idx[0] + 1} {times[idx[0]]} {idx[-1] + 1} {times[idx[-1]]}'
            print(info_str)
            f.write(info_str + "\n")
            f.flush()

            time_subset = times[mask]
            mag_subset  = mags[mask]
            fig, ax = plt.subplots()
            ax.scatter(time_subset, mag_subset)
            ax.invert_yaxis()
            # Save figure to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close(fig)
            with open(preview_file_name, "a") as f_preview:
                f_preview.write(f"<p>[{n}] {info_str}</p>\n")
                f_preview.write(f"<img src='data:image/png;base64,{encoded}'><br><br>\n")
                f_preview.write("<hr>\n")
                f_preview.flush()

with open(preview_file_name, "a") as f_preview:
    f_preview.write("<p>End of file</p>\n")
    f_preview.write("\n</body></html>")                
        