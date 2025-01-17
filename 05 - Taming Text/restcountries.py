# -*- coding: utf-8 -*-
import requests
import json
from future.utils import python_2_unicode_compatible

BASE_URI = 'https://restcountries.eu/rest/v1'


class RestCountryApi(object):

    @classmethod
    def _get_country_list(cls, resource, term=''):
        """Takes a resource and a search term and return a list of countries or a country.

        :param resource - resource to create the URL
        :param term - search term provided by the user of this package
        :returns - either a Country object or a list of Countries
        """
        uri = '{}{}/{}'.format(BASE_URI, resource, term)  # build URL
        response = requests.get(uri)
        if response.status_code == 200:
            result_list = []
            data = json.loads(response.text)  # parse json to dict
            if type(data) == list:
                for country_data in data:  # in case it is a list create python list with country instances
                    country = Country(country_data)
                    result_list.append(country)
            else:
                return Country(data)
            return result_list
        elif response.status_code == 404:
            raise requests.exceptions.InvalidURL
        else:
            raise requests.exceptions.RequestException

    @classmethod
    def get_all(cls):
        """Returns all countries provided by  restcountries.eu."""
        resource = '/all'
        return cls._get_country_list(resource)

    @classmethod
    def get_countries_by_name(cls, name):
        """Returns a list of countries.

        :param name - Name string of a country. E.g. 'France'.
        :returns: list of Country objects
        """
        resource = '/name'
        return cls._get_country_list(resource, name)

    @classmethod
    def get_countries_by_language(cls, language):
        """Returns a list of countries.

        :param language - Language string of a country. E.g. 'en'.
        :returns: list of Country objects
        """
        resource = '/lang'
        return cls._get_country_list(resource, language)

    @classmethod
    def get_countries_by_calling_code(cls, calling_code):
        """Returns a list of countries.

        :param calling_code - Calling code string of a country. E.g. '1'.
        :returns: list of Country objects
        """
        resource = '/callingcode'
        return cls._get_country_list(resource, calling_code)

    @classmethod
    def get_country_by_country_code(cls, alpha):
        """Returns a list of countries.

        :param alpha - Alpha code string of a country. E.g. 'de'.
        :returns: a Country object
        You can look those up at wikipedia: https://en.wikipedia.org/wiki/ISO_3166-1
        """
        resource = '/alpha'
        return cls._get_country_list(resource, alpha)

    @classmethod
    def get_country_by_country_codes(cls, codes):
        """Returns a list of countries.

        :param codes - List of strings which represent the codes of countries. E.g. ['us', 'uk']
        You can look those up at wikipedia: https://en.wikipedia.org/wiki/ISO_3166-1
        :returns: list of Country objects
        """
        resource = '/alpha?codes='
        codes = ','.join(codes)
        return cls._get_country_list(resource, codes)

    @classmethod
    def get_countries_by_curreny(cls, currency):
        """Returns a list of countries.

        :param currency - Currency string of a country. E.g. 'EUR'.
        :returns: list of Country objects
        """
        resource = '/currency'
        return cls._get_country_list(resource, currency)

    @classmethod
    def get_countries_by_region(cls, region):
        """Returns a list of countries.

        :param region - Region string of a country. E.g. 'Europe'.
        :returns: list of Country objects
        """
        resource = '/region'
        return cls._get_country_list(resource, region)

    @classmethod
    def get_countries_by_subregion(cls, subregion):
        """Returns a list of countries.

        :param subregion - Subregion string of a country. E.g. 'Western Europe'
        :returns: list of Country objects
        """
        resource = '/subregion'
        return cls._get_country_list(resource, subregion)

    @classmethod
    def get_countries_by_capital(cls, capital):
        """Returns a list of countries.

        :param capital - Capital string of a country. E.g. 'London'
        :returns: list of Country objects
        """
        resource = '/capital'
        return cls._get_country_list(resource, capital)


@python_2_unicode_compatible
class Country(object):

    def __str__(self):
        return u'{}'.format(self.name)

    def __init__(self, country_data):
        # print(country_data)
        self.top_level_domain = country_data['topLevelDomain']
        self.alpha2_code = country_data['alpha2Code']
        self.alpha3_code = country_data['alpha3Code']
        self.currencies = country_data['currencies']
        self.capital = country_data['capital']
        self.calling_codes = country_data['callingCodes']
        self.alt_spellings = country_data['altSpellings']
        self.relevance = country_data['relevance']
        self.region = country_data['region']
        self.subregion = country_data['subregion']
        self.translation = country_data['translations']
        self.population = country_data['population']
        self.latlng = country_data['latlng']
        self.demonym = country_data['demonym']
        self.area = country_data['area']
        self.gini = country_data['gini']
        self.timezones = country_data['timezones']
        self.borders = country_data['borders']
        self.native_name = country_data['nativeName']
        self.name = country_data['name']
