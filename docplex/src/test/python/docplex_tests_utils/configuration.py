'''
Created on Apr 16, 2015

@author: kong
'''
import os.path
import socket
try:
    from configparser import ConfigParser, NoOptionError, NoSectionError
except ImportError:
    from ConfigParser import ConfigParser, NoOptionError, NoSectionError
    
from six import iteritems

from docplex.mp.utils import get_logger


class CascadedConfiguration:
    """ A cascaded, simplified ConfigParser.
    
    This only implement a few getters for the ConfigParser.
    """
    BOOLEAN_STATES = {'1': True, 'yes': True, 'true': True, 'on': True,
                      '0': False, 'no': False, 'false': False, 'off': False}
    

    def __init__(self, parent=None, config_parser=None):
        """Creates a new Cascaded configuration.
        
        When an option is queried from this configuration, the config_parser
        is searched for this option. If the config_parser does not have this
        option, the parent configuration or config_parser is search for the
        option.
        
        Args:
            parent: The parent configuration.
            config_parser: The config parser containing the configuration.
        """
        self.parent = parent
        self.config_parser = config_parser
        
    @classmethod
    def create_from_ini(cls, basename, path=None, cascade=True):    
        """Creates a Cascaded configuration from hierarchical ini files.
            
        This is basically a ConfigParser with hierarchical capacity.
        
        When an option is queried, the option is first search in
        path/basename.hostname.ini if that file exists.
        Otherwise, this will look for the option in path/basename.ini
        
        Example:
            You have a config.ini containing::
            
                [DOcloudConnector]
                verbose = True
                url = https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/
    
            On your machine, you have a config.hostname.ini containing::
            
               [DOcloudConnector]
               verbose = False
                
            You initialize a CascadedConfiguration with::
            
                >>> config = CascadedConfiguration.create_from_ini("config", your_path)
                >>> config.getboolean("DOcloudConnector", "verbose")
                False
                >>> config.getvalue("DOcloudConnector", "url")
                https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/            
            
        Arguments:
            basename: The basename for the ``*``.ini files, without the .ini
            path: The path where to find the ``*``.ini
        """        
        base_config_name = basename
        if path is not None:
            base_config_name = os.path.join(path, base_config_name)
    
        logger = get_logger('CascadedConfiguration', True)
  
        # This is the default config file
        parser = ConfigParser()
        defaultConfigName = base_config_name + ".ini"
        # check that we find path/basename.ini. If yes, read that conifg
        if os.path.isfile(defaultConfigName):
            logger.info("Found config file " + defaultConfigName)
            parser.read(defaultConfigName)
        else:
            logger.info("Did not find config file " + defaultConfigName)
            
        defaultConfig = CascadedConfiguration(config_parser=parser)
        
        # in case we don't want the host specific file but just the global file
        if not cascade:
            return defaultConfig

        # This is to find the host specific config
        hostname = socket.gethostname()
        host_config_name = basename + "." + hostname
        if path is not None:
            host_config_name = os.path.join(path, host_config_name)
            
        hostParser = ConfigParser()
        hostConfigName = host_config_name + ".ini"       
        # check that we find path/basename.hostname.ini. If yes, read that conifg
        if os.path.isfile(hostConfigName):
            logger.info("Found host specific config file = " + hostConfigName)
            hostParser.read(hostConfigName)
        else:
            logger.info("Did not find config file " + defaultConfigName)
            
        hostConfig = CascadedConfiguration(parent=defaultConfig,
                                           config_parser=hostParser)

        return hostConfig
        
        
    def _convert_to_bool(self, value):
        if value.lower() not in self.BOOLEAN_STATES:
            raise ValueError('Not a boolean: %s' % value)
        return self.BOOLEAN_STATES[value.lower()]
       

    def get(self, section, option):
        """Get an option value for a given option.
        
        Returns:
            The option value or None if the option does not exists.
        """
        value = None
        try:
            value = self.config_parser.get(section, option)
        except (NoOptionError, NoSectionError):
            value = None
            
        # look up value in parents if it is not found here
        if value is None and self.parent is not None:
            try:
                value = self.parent.get(section, option)
            except (NoOptionError, NoSectionError):
                value = None
           
        return value
    
    def getboolean(self, section, option):
        """Get an option value as boolean for a given option.
        
        Returns:
            The option value as a boolean value or None if the option does
            not exists.
        """
        value = self.get(section, option)
        if value is not None:
            return self._convert_to_bool(value)
        return None
     
    def getint(self, section, option):
        """Get an option value as int for a given option.
        
        Returns:
            The option value as an int value or None if the option does not
            exists.
        """
        value = self.get(section, option)
        if value is not None:
            return int(value)
        return value
    
    def options(self, section_name):
        """ Return a list of options for a given section.
        
        That list of option results from the merge of default config and
        host specific config.
        
        Returns:
            A list containing the options.
        """
        parentOptions = None
        if self.parent is not None:
            try:
                parentOptions = self.parent.options(section_name)
            except NoSectionError:
                parentOptions = None
                
        try:
            thisOptions = self.config_parser.options(section_name)
        except NoSectionError:
            thisOptions = None
        # merge list of options
        result = set()
        if parentOptions is not None:
            result |= set(parentOptions)
        if thisOptions is not None:
            result |= set(thisOptions)
        return list(result)

    def items(self, section_name):
        """ Return a list of (name,option) for a given section.
        
        That list of option results from the merge of default config and
        host specific config.
        
        Returns:
            A list containing the options.
        """
        parentOptions = None
        if self.parent is not None:
            try:
                parentOptions = self.parent.items(section_name)
            except NoSectionError:
                parentOptions = None
                
        try:
            thisOptions = self.config_parser.items(section_name)
        except NoSectionError:
            thisOptions = None
        # build dict with parent options
        result = {}
        if parentOptions is not None:
            for (k,v) in parentOptions:
                result[k] = v
        # override/add this options
        if thisOptions is not None:
            for (k,v) in thisOptions:
                result[k] = v
        # create list
        result_list = []
        for k,v in iteritems(result):
            result_list.append((k,v))
        return result_list
