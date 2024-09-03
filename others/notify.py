import socket

import requests


def notify(message):
    url = 'https://hooks.slack.com/services/' \
          'T4HV8KJA3/BPC1N86M9/156MGIVayM8yXyxcd5slwaz6'
    message = f'{message} at {socket.gethostname()}.'
    requests.post(url=url, data=f'{{"text": "{message}"}}')


def main():
    notify('Finished')


if __name__ == '__main__':
    main()
