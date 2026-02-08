from dotenv import load_dotenv
from config.settings import settings
from pycti import OpenCTIApiClient
def main():
    load_dotenv()
    client = OpenCTIApiClient(settings.opencti_url, settings.opencti_token)
    print("Connected to OpenCTI")
    malw = client.malware.create(name="ExampleMalware", is_family=False, description="Mock malware", confidence=40)
    ind1 = client.indicator.create(name="IOC-Alpha", pattern_type="stix", pattern="[ipv4-addr:value = '10.0.0.5']", confidence=70)
    ind2 = client.indicator.create(name="IOC-Beta",  pattern_type="stix", pattern="[ipv4-addr:value = '10.0.0.6']", confidence=60)
    client.stix_core_relationship.create(relationship_type="indicates", fromId=ind1["id"], toId=malw["id"], confidence=65)
    client.stix_core_relationship.create(relationship_type="indicates", fromId=ind2["id"], toId=malw["id"], confidence=55)
    print("Seeded mock data. IDs:", malw["id"], ind1["id"], ind2["id"])
if __name__ == "__main__": main()
