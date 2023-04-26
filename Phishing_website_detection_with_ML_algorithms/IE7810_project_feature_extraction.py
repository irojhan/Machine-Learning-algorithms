import pandas as pd
from urllib.parse import urlparse,urlencode
import ipaddress
import re
from bs4 import BeautifulSoup
import whois
import urllib
import urllib.request
from datetime import datetime
import requests
#Downloading the phishing URLs file
!wget http://data.phishtank.com/data/online-valid.csv
#loading the phishing URLs data to dataframe
dfph = pd.read_csv("online-valid.csv")
#Collecting 5,000 Phishing URLs randomly
phishing_url = dfph.sample(n = 5000, random_state = 12).copy()
phishing_url = phishing_url.reset_index(drop=True)
#Loading legitimate files
dfl = pd.read_csv("Benign_list_big_final.csv")
dfl.columns = ['URLs']
#Collecting 5,000 Legitimate URLs randomly
legitimate_url = dfl.sample(n = 5000, random_state = 12).copy()
legitimate_url = legitimate_url.reset_index(drop=True)


                                ### Feature Extraction ###

# IP address
def IP_check(url):
  try:
    ipaddress.ip_address(url)
    ip = 1
  except:
    ip = 0
  return ip

# URL length
def getLength(url):
  if len(url) < 54:
    length = 0
  else:
    length = 1
  return length

# @ check
def AtSign(url):
  if "@" in url:
    at = 1
  else:
    at = 0
  return at

from urllib.parse import urlparse

def extract_domain(url):
    domain = urlparse(url).netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

# URL depth
def get_url_depth(url):
    path_parts = urlparse(url).path.split('/')
    depth = sum(1 for part in path_parts if part)
    return depth

# redirection check
def has_redirection(url):
    pos = url.rfind('//')
    return pos > 6 and pos != 7

# Suffix prefix check
def has_prefix_suffix(url):
    return '-' in urlparse(url).netloc


#HTTPS check
def has_https(url):
    return urlparse(url).scheme == "https"

#tiny URLs
#listing shortening services
SHORTENING_SERVICES_PATTERN = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                      r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                      r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                      r"tr\.im|link\.zip\.net"


def tinyURL(url):
    match=re.search(SHORTENING_SERVICES_PATTERN,url)
    if match:
        return 1
    else:
        return 0

# web traffic
def is_web_traffic_legitimate(url):
    """
    Check if the website has legitimate web traffic using Alexa traffic rank.
    Returns 1 if web traffic is not legitimate, and 0 if it is.
    """
    try:
        # Fill the whitespaces in the URL if any
        url = urllib.parse.quote(url)
        rank = BeautifulSoup(urllib.request.urlopen(f"http://data.alexa.com/data?cli=10&dat=s&url={url}").read(), "xml") \
            .find("REACH")['RANK']
        rank = int(rank)
    except (TypeError, AttributeError):
        return 1
    if rank < 100000:
        return 1
    else:
        return 0


# domain age
from datetime import datetime

def domain_age(domain):
    creation_date = domain.creation_date
    expiration_date = domain.expiration_date

    if isinstance(creation_date, str) or isinstance(expiration_date, str):
        try:
            creation_date = datetime.strptime(creation_date, '%Y-%m-%d')
            expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')
        except:
            return 1

    if expiration_date is None or creation_date is None:
        return 1

    if type(expiration_date) is list or type(creation_date) is list:
        return 1

    age_of_domain = abs((expiration_date - creation_date).days)

    if age_of_domain / 30 < 6:
        return 1  # phishing
    else:
        return 0  # legitimate


#domain end period
def domainEnd(domain_name):
    expiration_date = domain_name.expiration_date
    if not isinstance(expiration_date, datetime):
        try:
            expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        except:
            return 1
    today = datetime.now()
    days_to_expire = (expiration_date - today).days
    if days_to_expire < 0:
        return 1  # domain has already expired
    elif days_to_expire < 180:
        return 1  # domain will expire in less than 6 months
    else:
        return 0  # domain will expire in more than 6 months


#IFrame redirection
def iframe(response):
  if response == "":
      return 1
  else:
      if re.findall(r"<iframe|<frameBorder", response.text):
          return 0
      else:
          return 1


# MouseOver check
def mouseOver(response):
    if not response:
        return 1  # Empty response

    # Find all <script> tags that include "onmouseover"
    scripts = response.find_all('script', text=re.compile(r'onmouseover', re.IGNORECASE))

    return 1 if scripts else 0  # Return 1 if there are any, otherwise return 0


#right click
def rightClick(response):
  if response == "":
    return 1
  else:
    if re.findall(r"oncontextmenu\s*=\s*['\"]return false['\"]", response.text):
      return 0
    else:
      return 1


# website forwarding
def forwarding(response):
    if not response:
        return 1
    if len(response.history) <= 2:
        return 0
    return 1

# feature extraction function

def extract_features(url, label):
    # Address bar-based features (10)
    domain = extract_domain(url)
    ip = IP_check(domain)
    at_sign = AtSign(url)
    length = getLength(url)
    depth = get_url_depth(url)
    redirection = has_redirection(url)
    https = has_https(url)
    tiny_url = tinyURL(url)
    prefix_suffix = has_prefix_suffix(url)

    # Domain-based features (4)
    dns_error = 0
    age_error = 0
    end_error = 0
    try:
        domain_info = whois.whois(domain)
        age = domain_age(domain_info)
        end = domainEnd(domain_info)
    except:
        dns_error = 1
        age_error = 1
        end_error = 1
        age = 1
        end = 1

    traffic_legitimate = is_web_traffic_legitimate(url)

    # HTML & Javascript-based features (4)
    try:
        response = requests.get(url)
        has_iframe = iframe(response)
        has_mouseover = mouseOver(response)
        has_rightclick = rightClick(response)
        has_forwarding = forwarding(response)
    except:
        has_iframe = 1
        has_mouseover = 1
        has_rightclick = 1
        has_forwarding = 1

    # Combine all features into a list
    features = [domain, ip, at_sign, length, depth, redirection, https, tiny_url, prefix_suffix,
                dns_error, traffic_legitimate, age_error, age, end_error, end,
                has_iframe, has_mouseover, has_rightclick, has_forwarding, label]

    return features


#Extracting the feautres & storing them in a list
legi_features = []
label = 0

for i in range(0, 5000):
    url = legitimate_url['URLs'][i]
    try:
        features = extract_features(url, label)
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        continue

    legi_features.append(features)


#converting the list to dataframe
feature_names = ['Domain', 'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection',
                      'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic',
                      'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards', 'Label']
legitimate = pd.DataFrame(legi_features, columns= feature_names)
legitimate.to_csv('legitimate.csv', index= False)

# Extract features for each URL and append to phish_features list
phish_features = []
label = 1
for i in range(len(phishing_url)):
  url = phishing_url.iloc[i]['url']
  phish_features.append(extract_features(url, label))

#converting the list to dataframe
feature_names = ['Domain', 'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection',
                      'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic',
                      'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards', 'Label']
phishing = pd.DataFrame(phish_features, columns= feature_names)
phishing.to_csv('phishing.csv', index= False)
#Concatenating the dataframes into one
urldata = pd.concat([legitimate, phishing]).reset_index(drop=True)
# Storing the data in CSV file
urldata.to_csv('urldata.csv', index=False)