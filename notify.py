import socket

import requests


def notify():
    url = 'https://hooks.slack.com/services/T4HV8KJA3/BPC1N86M9/' \
          '156MGIVayM8yXyxcd5slwaz6'
    msg = 'Experiments done at {}.'.format(socket.gethostname())
    requests.post(url=url, data='{{"text": "{}"}}'.format(msg))


if __name__ == '__main__':
    notify()
