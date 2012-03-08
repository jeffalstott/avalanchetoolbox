#Some code, particularly how to join up the Fit table with the AvalancheAnalysis with a polymorphic association, taken from http://techspot.zzzeek.org/files/2007/discriminator_on_association.py
#This polymorphic association was set up in order to allow for future, different kinds of analyses that also would warrant distribution fit analyses
from sqlalchemy import Column, Float, Integer, String, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.associationproxy import association_proxy

class Base(object):
    """Base class which provides automated table name
    and surrogate primary key column.
    """
    @declared_attr
    def __tablename__(cls):
        return cls.__name__
    id = Column(Integer, primary_key=True)

    def __repr__(self):
        return str(vars(self))

Base = declarative_base(cls=Base)

class Subject(Base):
    species = Column(String(100))
    name = Column(String(100))
    group_name = Column(String(100))
    number_in_group = Column(Integer)

class Sensor(Base):
    location = Column(String(100))
    sensor_type = Column(String(100))
    sensor_count = Column(Integer)
    sensors_locations_file = Column(String(100))
    sensors_spacing = Column(Float)

class Channel(Base):
    number = Column(Integer)
    name = Column(String(100))

    sensor_id = Column(Integer, ForeignKey('Sensor.id'))
    sensor = relationship(Sensor, cascade="all, delete-orphan", backref=backref('channels'), single_parent=True)

class Task(Base):
    type = Column(String(100))
    description = Column(String(100))
    eyes = Column(String(100))

class Experiment(Base):
    location = Column(String(100))
    date = Column(String(100))
    visit_number = Column(Integer)
    mains = Column(Integer)
    drug = Column(String(100))
    rest = Column(String(100))

    subject_id = Column(Integer, ForeignKey('Subject.id'))
    subject = relationship(Subject, cascade="all, delete-orphan", backref=backref('experiments'), single_parent=True)

    task_id = Column(Integer, ForeignKey('Task.id'))
    task = relationship(Task, cascade="all, delete-orphan", backref=backref('experiments'), single_parent=True)

class Task_Performance(Base):
    measure1_name = Column(String(100)) 
    measure1_value = Column(Float) 
    measure2_name = Column(String(100)) 
    measure2_value = Column(Float) 
    measure3_name = Column(String(100))
    measure3_value = Column(Float)
    measure4_name = Column(String(100)) 
    measure4_value = Column(Float) 
    measure5_name = Column(String(100)) 
    measure5_value = Column(Float) 
    measure6_name = Column(String(100))
    measure6_value = Column(Float)
    measure7_name = Column(String(100)) 
    measure7_value = Column(Float) 
    measure8_name = Column(String(100)) 
    measure8_value = Column(Float) 
    measure9_name = Column(String(100))
    measure9_value = Column(Float)
    measure10_name = Column(String(100))
    measure10_value = Column(Float)

    task_id = Column(Integer, ForeignKey('Task.id'))
    task = relationship(Task, cascade="all, delete-orphan", backref=backref('task_performances'), single_parent=True)

    experiment_id = Column(Integer, ForeignKey('Experiment.id'))
    experiment = relationship(Experiment, cascade="all, delete-orphan", backref=backref('task_performances'), single_parent=True)

class Recording(Base):
    duration = Column(Float)
    sampling_rate = Column(Float)
    maxfilter = Column(Boolean)
    transd = Column(Boolean)
    eye_movement_removed = Column(Boolean)

    experiment_id = Column(Integer, ForeignKey('Experiment.id'))
    experiment = relationship(Experiment, cascade="all, delete-orphan", backref=backref('recordings'), single_parent=True)

    sensor_id = Column(Integer, ForeignKey('Sensor.id'))
    sensor = relationship(Sensor, cascade="all, delete-orphan", backref=backref('recordings'), single_parent=True)

class Filter(Base):
    filter_type = Column(String(100))
    poles = Column(Integer)
    window = Column(String(100))
    band_name = Column(String(100))
    band_min = Column(Float)
    band_max = Column(Float)
    duration = Column(Float)
    downsampled_rate = Column(Float)
    notch = Column(Boolean)
    phase_shuffled = Column(Boolean)

    recording_id = Column(Integer, ForeignKey('Recording.id'))
    recording = relationship(Recording, cascade="all, delete-orphan", backref=backref('filters'), single_parent=True)

class Threshold(Base):
    signal = Column(String(100))
    mode = Column(String(100))
    level = Column(Float)
    up = Column(Float)
    down = Column(Float)
    mean = Column(Float)

    channel = Column(Float)
    #channel_id = Column(Integer, ForeignKey('Channel.id'))
    #channel = relationship(Channel, cascade="all, delete-orphan", backref=backref('thresholds'), single_parent=True)

    filter_id = Column(Integer, ForeignKey('Filter.id'))
    filter = relationship(Filter, cascade="all, delete-orphan", backref=backref('thresholds'), single_parent=True)

class Event(Base):
    time = Column(Integer)
    interval = Column(Integer)
    detection = Column(String(100))
    direction = Column(String(100))
    displacement = Column(Float)
    amplitude = Column(Float)
    amplitude_auc = Column(Float)
    displacement_auc = Column(Float)
    t_ratio_amplitude = Column(Float)
    t_ratio_displacement = Column(Float)
    t_ratio_displacement_aucs = Column(Float)
    t_ratio_amplitude_aucs = Column(Float)

    #channel_id = Column(Integer, ForeignKey('Channel.id'))
    #channel = relationship(Channel, cascade="all, delete-orphan", backref=backref('events'), single_parent=True)

    threshold_id = Column(Integer, ForeignKey('Threshold.id'))
    filter = relationship(Threshold, cascade="all, delete-orphan", backref=backref('events'), single_parent=True)

class Fit_Association(Base):
    """Associates a collection of Fit objects
    with a particular analysis.
    """

    @classmethod
    def creator(cls, discriminator):
        """Provide a 'creator' function to use with 
        the association proxy."""

        return lambda fits:Fit_Association(
                                fits=fits, 
                                discriminator=discriminator)

    discriminator = Column(String(100))
    """Refers to the type of analysis."""

    @property
    def analysis(self):
        """Return the analysis object."""
        return getattr(self, "%s_analysis" % self.discriminator)

class HasFits(object):
    """HasFits mixin, creates a relationship to
    the address_association table for each parent.
    
    """
    @declared_attr
    def fit_association_id(cls):
        return Column(Integer, ForeignKey("Fit_Association.id"))

    @declared_attr
    def fit_association(cls):
        discriminator = cls.__name__.lower()
        cls.fits= association_proxy(
                    "fit_association", "fits",
                    creator=Fit_Association.creator(discriminator)
                )
        return relationship(Fit_Association, 
                    cascade="all, delete-orphan", backref=backref("%s_analysis" % discriminator, 
                                        uselist=False), single_parent=True)

class AvalancheAnalysis(HasFits,Base):
    spatial_sample = Column(String(100))
    temporal_sample = Column(String(100))
    threshold_mode = Column(String(100))
    threshold_level = Column(Float)
    threshold_direction = Column(String(100))
    time_scale = Column(Float)
    event_signal = Column(String(100))
    event_detection = Column(String(100))
    cascade_method = Column(String(100))

    n = Column(Integer)

    interevent_intervals_mean = Column(Float)
    interevent_intervals_median = Column(Float)
    interevent_intervals_mode = Column(Float)

    sigma_events = Column(Float)
    sigma_displacements = Column(Float)
    sigma_amplitudes = Column(Float)
    sigma_amplitude_aucs = Column(Float)

#    t_ratio_displacements_slope = Column(Float)
#    t_ratio_displacements_R = Column(Float)
#    t_ratio_displacements_p = Column(Float)
#    t_ratio_amplitudes_slope = Column(Float)
#    t_ratio_amplitudes_R = Column(Float)
#    t_ratio_amplitudes_p = Column(Float)
#    t_ratio_displacement_aucs_slope = Column(Float)
#    t_ratio_displacement_aucs_R = Column(Float)
#    t_ratio_displacement_aucs_p = Column(Float)
#    t_ratio_amplitude_aucs_slope = Column(Float)
#    t_ratio_amplitude_aucs_R = Column(Float)
#    t_ratio_amplitude_aucs_p = Column(Float)

    filter_id = Column(Integer, ForeignKey('Filter.id'))
    filter = relationship(Filter, cascade="all, delete-orphan", backref=backref('avalancheanalyses'), single_parent=True)

class Fit(Base):
    analysis_type = Column(String(100)) 
    variable = Column(String(100)) 
    method = Column(String(100))
    distribution = Column(String(100)) 
    parameter1_name = Column(String(100)) 
    parameter1 = Column(Float) 
    parameter2_name = Column(String(100)) 
    parameter2 = Column(Float) 
    parameter3_name = Column(String(100))
    parameter3 = Column(Float)
    fixed_xmin = Column(Boolean)
    xmin = Column(Float) 
    fixed_xmax = Column(Boolean)
    xmax = Column(Float)
    loglikelihood = Column(Float) 
    power_law_loglikelihood_ratio = Column(Float) 
    power_law_p = Column(Float)
    truncated_power_law_loglikelihood_ratio = Column(Float) 
    truncated_power_law_p = Column(Float)
    D = Column(Float)
    D_plus_critical_branching = Column(Float)
    D_minus_critical_branching = Column(Float)
    Kappa = Column(Float)
#    p = Column(Float)
    n_tail = Column(Integer)
    noise_flag = Column(Boolean)
    discrete = Column(Boolean)

    analysis_id = Column(Integer)
    analysis = association_proxy("association", "analysis")
    association_id = Column(Integer, ForeignKey("Fit_Association.id"))
    association = relationship(Fit_Association, cascade="all, delete-orphan", backref=backref("fits"), single_parent=True)

class Avalanche(Base):
    duration = Column(Integer)
    size_events = Column(Integer)
    size_displacements = Column(Float)
    size_amplitudes = Column(Float)
    size_amplitude_aucs = Column(Float)
    sigma_events = Column(Float)
    sigma_displacements = Column(Float)
    sigma_amplitudes = Column(Float)
    sigma_displacement_aucs = Column(Float)
    sigma_amplitude_aucs = Column(Float)

    analysis_id = Column(Integer, ForeignKey('AvalancheAnalysis.id'))
    analysis = relationship(AvalancheAnalysis, cascade="all, delete-orphan", backref=backref('avalanches'), single_parent=True)

def create_database(url):
    from sqlalchemy import create_engine

    engine = create_engine(url, echo=False)
    Base.metadata.create_all(engine)

def compare(session, *args, **kwargs):
    """compare does things"""

    data = session.query(*args).\
        join(Fit_Association).\
        join(AvalancheAnalysis, AvalancheAnalysis.id==Fit_Association.id).\
        join(Filter, Filter.id==Fit.filter_id).\
        join(Recording, Recording.id==Fit.recording_id).\
        join(Experiment, Experiment.id==Fit.experiment_id).\
        join(Sensor, Sensor.id==Fit.sensor_id).\
        join(Subject, Subject.id==Fit.subject_id).\
        join(Task, Task.id==Fit.task_id)

    filters = {
        'Sensor.sensor_type': 'gradiometer',\
        'Task.eyes': 'open',\
        'Experiment.visit_number': None,\
        'Filter.band_name': 'broad',\
        'Filter.downsampled_rate': '1000',\
        'Subject.group_name': None,\
        'AvalancheAnalysis.spatial_sample': 'all',\
        'AvalancheAnalysis.temporal_sample': 'all',\
        'AvalancheAnalysis.threshold_mode': 'SD',\
        'AvalancheAnalysis.threshold_level': 3,\
        'AvalancheAnalysis.threshold_direction': 'both',\
        'AvalancheAnalysis.event_signal': 'displacement',\
        'AvalancheAnalysis.event_detection': 'local_extrema',\
        'AvalancheAnalysis.cascade_method': 'grid',\
        'Fit.analysis_type':  'avalanches',\
        'Fit.variable':  'size_events',\
        'Fit.distribution':  'power_law',\
        'Fit.fixed_xmin':  True,\
        'Fit.xmin': 1,\
        'Fit.fixed_xmax':  True,\
        'Fit.xmax': 204,\
        'AvalancheAnalysis.time_scale': 2,\
        }

    filters.update(kwargs)
    print filters

    for key in filters:
        if filters[key]==None:
            continue
        table, variable = key.split('.')
        if type(filters[key])==tuple:
            if type(filters[key][0])==tuple:
                f1 = filters[key][0]
                f2 = filters[key][-1]
                data = data.filter(f1[-1](getattr(getattr(database_classes,table),variable), f1[0]))
                data = data.filter(f2[-1](getattr(getattr(database_classes,table),variable), f2[0]))
            else:
                data = data.filter(filters[key][-1](getattr(getattr(database_classes,table),variable), filters[key][0]))
        else:
            data = data.filter(getattr(getattr(database_classes,table),variable)==filters[key])

    d = data.all()
    session.close()
    session.bind.dispose()
    types = []
    for a in range(len(args)):
        name = str(args[a]).split('.')[-1]
        t = type(d[0][a])
        if t==str:
            t = '|S100'
        types.append((name, t))
    from numpy import array
    return array(d, dtype = types) 
