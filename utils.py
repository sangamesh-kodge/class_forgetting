
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from collections import  Counter
from collections import OrderedDict
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import os
from models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.vit import vit_b_16, vit_b_32,  vit_l_16, vit_l_32, vit_h_14
from torchvision import datasets, transforms
from collections import defaultdict
from sklearn.svm import SVC
import os
from torch.utils.data import Dataset

np.set_printoptions(suppress=True)

def get_dataset(args):
    if "fmnist" in args.dataset:
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
            ])
        
        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
            ])
        if args.dataset =="fmnist":
            if args.train_transform:
                dataset1 = datasets.FashionMNIST(args.data_path, train=True, download=True, transform=train_transform)
            else:
                dataset1 = datasets.FashionMNIST(args.data_path, train=True, download=True, transform=test_transform)
            dataset2 = datasets.FashionMNIST(args.data_path, train=False, transform=test_transform)
            args.num_classes = 10
            args.class_label_names = [i for i in range(10)]
        else:
            raise ValueError 
    elif "mnist" in args.dataset:
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        if args.dataset =="mnist":
            if args.train_transform:
                dataset1 = datasets.MNIST(args.data_path, train=True, download=True, transform=train_transform)
            else:
                dataset1 = datasets.MNIST(args.data_path, train=True, download=True, transform=test_transform)
            dataset2 = datasets.MNIST(args.data_path, train=False, transform=test_transform)
            args.num_classes = 10
            args.class_label_names = [i for i in range(10)]
        else:
            raise ValueError 
    elif "cifar" in args.dataset:
        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                        std=[0.247, 0.243, 0.262]),      
            ])
        
        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                        std=[0.247, 0.243, 0.262]),   
            ])
        if args.dataset == "cifar10":
            if args.train_transform:
                dataset1 = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
            else:
                dataset1 = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=test_transform)
            dataset2 = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
            args.num_classes=10
            args.class_label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        elif args.dataset == "cifar100":
            if args.train_transform:
                dataset1 = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=train_transform)
            else:
                dataset1 = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=test_transform)
            dataset2 = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=test_transform)
            args.num_classes=100
            args.class_label_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        else:
            raise ValueError
    elif "imagenet" in args.dataset:
        train_transform=transforms.Compose([ 
            transforms.RandomResizedCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),    
            ])

        test_transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])     
            ])
        if args.dataset == "imagenette":
            if args.train_transform:
                dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "imagenette","train"), transform=train_transform)
            else:
                dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "imagenette","train"), transform=test_transform)
            dataset2 = datasets.ImageFolder(root=os.path.join(args.data_path, "imagenette","val"), transform=test_transform)
            args.num_classes = 10
            args.class_label_names = ["bench", "English springer", "cassette player", "chain saw", "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
        elif args.dataset == "imagenet":
            if args.train_transform:
                dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=train_transform)
            else:
                dataset1 = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=test_transform)
            dataset2 = datasets.ImageFolder(root=os.path.join(args.data_path, "val"), transform=test_transform)
            args.num_classes = 1000
            args.class_label_names = ['tench, Tinca tinca', 'goldfish, Carassius auratus', 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 'tiger shark, Galeocerdo cuvieri', 'hammerhead, hammerhead shark', 'electric ray, crampfish, numbfish, torpedo', 'stingray', 'cock', 'hen', 'ostrich, Struthio camelus', 'brambling, Fringilla montifringilla', 'goldfinch, Carduelis carduelis', 'house finch, linnet, Carpodacus mexicanus', 'junco, snowbird', 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'robin, American robin, Turdus migratorius', 'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel, dipper', 'kite', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'vulture', 'great grey owl, great gray owl, Strix nebulosa', 'European fire salamander, Salamandra salamandra', 'common newt, Triturus vulgaris', 'eft', 'spotted salamander, Ambystoma maculatum', 'axolotl, mud puppy, Ambystoma mexicanum', 'bullfrog, Rana catesbeiana', 'tree frog, tree-frog', 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui', 'loggerhead, loggerhead turtle, Caretta caretta', 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 'mud turtle', 'terrapin', 'box turtle, box tortoise', 'banded gecko', 'common iguana, iguana, Iguana iguana', 'American chameleon, anole, Anolis carolinensis', 'whiptail, whiptail lizard', 'agama', 'frilled lizard, Chlamydosaurus kingi', 'alligator lizard', 'Gila monster, Heloderma suspectum', 'green lizard, Lacerta viridis', 'African chameleon, Chamaeleo chamaeleon', 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis', 'African crocodile, Nile crocodile, Crocodylus niloticus', 'American alligator, Alligator mississipiensis', 'triceratops', 'thunder snake, worm snake, Carphophis amoenus', 'ringneck snake, ring-necked snake, ring snake', 'hognose snake, puff adder, sand viper', 'green snake, grass snake', 'king snake, kingsnake', 'garter snake, grass snake', 'water snake', 'vine snake', 'night snake, Hypsiglena torquata', 'boa constrictor, Constrictor constrictor', 'rock python, rock snake, Python sebae', 'Indian cobra, Naja naja', 'green mamba', 'sea snake', 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus', 'diamondback, diamondback rattlesnake, Crotalus adamanteus', 'sidewinder, horned rattlesnake, Crotalus cerastes', 'trilobite', 'harvestman, daddy longlegs, Phalangium opilio', 'scorpion', 'black and gold garden spider, Argiope aurantia', 'barn spider, Araneus cavaticus', 'garden spider, Aranea diademata', 'black widow, Latrodectus mactans', 'tarantula', 'wolf spider, hunting spider', 'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse, partridge, Bonasa umbellus', 'prairie chicken, prairie grouse, prairie fowl', 'peacock', 'quail', 'partridge', 'African grey, African gray, Psittacus erithacus', 'macaw', 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser, Mergus serrator', 'goose', 'black swan, Cygnus atratus', 'tusker', 'echidna, spiny anteater, anteater', 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus', 'wallaby, brush kangaroo', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'wombat', 'jellyfish', 'sea anemone, anemone', 'brain coral', 'flatworm, platyhelminth', 'nematode, nematode worm, roundworm', 'conch', 'snail', 'slug', 'sea slug, nudibranch', 'chiton, coat-of-mail shell, sea cradle, polyplacophore', 'chambered nautilus, pearly nautilus, nautilus', 'Dungeness crab, Cancer magister', 'rock crab, Cancer irroratus', 'fiddler crab', 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica', 'American lobster, Northern lobster, Maine lobster, Homarus americanus', 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 'crayfish, crawfish, crawdad, crawdaddy', 'hermit crab', 'isopod', 'white stork, Ciconia ciconia', 'black stork, Ciconia nigra', 'spoonbill', 'flamingo', 'little blue heron, Egretta caerulea', 'American egret, great white heron, Egretta albus', 'bittern', 'crane', 'limpkin, Aramus pictus', 'European gallinule, Porphyrio porphyrio', 'American coot, marsh hen, mud hen, water hen, Fulica americana', 'bustard', 'ruddy turnstone, Arenaria interpres', 'red-backed sandpiper, dunlin, Erolia alpina', 'redshank, Tringa totanus', 'dowitcher', 'oystercatcher, oyster catcher', 'pelican', 'king penguin, Aptenodytes patagonica', 'albatross, mollymawk', 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus', 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca', 'dugong, Dugong dugon', 'sea lion', 'Chihuahua', 'Japanese spaniel', 'Maltese dog, Maltese terrier, Maltese', 'Pekinese, Pekingese, Peke', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 'toy terrier', 'Rhodesian ridgeback', 'Afghan hound, Afghan', 'basset, basset hound', 'beagle', 'bloodhound, sleuthhound', 'bluetick', 'black-and-tan coonhound', 'Walker hound, Walker foxhound', 'English foxhound', 'redbone', 'borzoi, Russian wolfhound', 'Irish wolfhound', 'Italian greyhound', 'whippet', 'Ibizan hound, Ibizan Podenco', 'Norwegian elkhound, elkhound', 'otterhound, otter hound', 'Saluki, gazelle hound', 'Scottish deerhound, deerhound', 'Weimaraner', 'Staffordshire bullterrier, Staffordshire bull terrier', 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 'Lakeland terrier', 'Sealyham terrier, Sealyham', 'Airedale, Airedale terrier', 'cairn, cairn terrier', 'Australian terrier', 'Dandie Dinmont, Dandie Dinmont terrier', 'Boston bull, Boston terrier', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer', 'Scotch terrier, Scottish terrier, Scottie', 'Tibetan terrier, chrysanthemum dog', 'silky terrier, Sydney silky', 'soft-coated wheaten terrier', 'West Highland white terrier', 'Lhasa, Lhasa apso', 'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer', 'vizsla, Hungarian pointer', 'English setter', 'Irish setter, red setter', 'Gordon setter', 'Brittany spaniel', 'clumber, clumber spaniel', 'English springer, English springer spaniel', 'Welsh springer spaniel', 'cocker spaniel, English cocker spaniel, cocker', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog, bobtail', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'collie', 'Border collie', 'Bouvier des Flandres, Bouviers des Flandres', 'Rottweiler', 'German shepherd, German shepherd dog, German police dog, alsatian', 'Doberman, Doberman pinscher', 'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard, St Bernard', 'Eskimo dog, husky', 'malamute, malemute, Alaskan malamute', 'Siberian husky', 'dalmatian, coach dog, carriage dog', 'affenpinscher, monkey pinscher, monkey dog', 'basenji', 'pug, pug-dog', 'Leonberg', 'Newfoundland, Newfoundland dog', 'Great Pyrenees', 'Samoyed, Samoyede', 'Pomeranian', 'chow, chow chow', 'keeshond', 'Brabancon griffon', 'Pembroke, Pembroke Welsh corgi', 'Cardigan, Cardigan Welsh corgi', 'toy poodle', 'miniature poodle', 'standard poodle', 'Mexican hairless', 'timber wolf, grey wolf, gray wolf, Canis lupus', 'white wolf, Arctic wolf, Canis lupus tundrarum', 'red wolf, maned wolf, Canis rufus, Canis niger', 'coyote, prairie wolf, brush wolf, Canis latrans', 'dingo, warrigal, warragal, Canis dingo', 'dhole, Cuon alpinus', 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 'hyena, hyaena', 'red fox, Vulpes vulpes', 'kit fox, Vulpes macrotis', 'Arctic fox, white fox, Alopex lagopus', 'grey fox, gray fox, Urocyon cinereoargenteus', 'tabby, tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat, Siamese', 'Egyptian cat', 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'lynx, catamount', 'leopard, Panthera pardus', 'snow leopard, ounce, Panthera uncia', 'jaguar, panther, Panthera onca, Felis onca', 'lion, king of beasts, Panthera leo', 'tiger, Panthera tigris', 'cheetah, chetah, Acinonyx jubatus', 'brown bear, bruin, Ursus arctos', 'American black bear, black bear, Ursus americanus, Euarctos americanus', 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'sloth bear, Melursus ursinus, Ursus ursinus', 'mongoose', 'meerkat, mierkat', 'tiger beetle', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'ground beetle, carabid beetle', 'long-horned beetle, longicorn, longicorn beetle', 'leaf beetle, chrysomelid', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant, emmet, pismire', 'grasshopper, hopper', 'cricket', 'walking stick, walkingstick, stick insect', 'cockroach, roach', 'mantis, mantid', 'cicada, cicala', 'leafhopper', 'lacewing, lacewing fly', "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk", 'damselfly', 'admiral', 'ringlet, ringlet butterfly', 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'cabbage butterfly', 'sulphur butterfly, sulfur butterfly', 'lycaenid, lycaenid butterfly', 'starfish, sea star', 'sea urchin', 'sea cucumber, holothurian', 'wood rabbit, cottontail, cottontail rabbit', 'hare', 'Angora, Angora rabbit', 'hamster', 'porcupine, hedgehog', 'fox squirrel, eastern fox squirrel, Sciurus niger', 'marmot', 'beaver', 'guinea pig, Cavia cobaya', 'sorrel', 'zebra', 'hog, pig, grunter, squealer, Sus scrofa', 'wild boar, boar, Sus scrofa', 'warthog', 'hippopotamus, hippo, river horse, Hippopotamus amphibius', 'ox', 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis', 'bison', 'ram, tup', 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis', 'ibex, Capra ibex', 'hartebeest', 'impala, Aepyceros melampus', 'gazelle', 'Arabian camel, dromedary, Camelus dromedarius', 'llama', 'weasel', 'mink', 'polecat, fitch, foulmart, foumart, Mustela putorius', 'black-footed ferret, ferret, Mustela nigripes', 'otter', 'skunk, polecat, wood pussy', 'badger', 'armadillo', 'three-toed sloth, ai, Bradypus tridactylus', 'orangutan, orang, orangutang, Pongo pygmaeus', 'gorilla, Gorilla gorilla', 'chimpanzee, chimp, Pan troglodytes', 'gibbon, Hylobates lar', 'siamang, Hylobates syndactylus, Symphalangus syndactylus', 'guenon, guenon monkey', 'patas, hussar monkey, Erythrocebus patas', 'baboon', 'macaque', 'langur', 'colobus, colobus monkey', 'proboscis monkey, Nasalis larvatus', 'marmoset', 'capuchin, ringtail, Cebus capucinus', 'howler monkey, howler', 'titi, titi monkey', 'spider monkey, Ateles geoffroyi', 'squirrel monkey, Saimiri sciureus', 'Madagascar cat, ring-tailed lemur, Lemur catta', 'indri, indris, Indri indri, Indri brevicaudatus', 'Indian elephant, Elephas maximus', 'African elephant, Loxodonta africana', 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens', 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', 'barracouta, snoek', 'eel', 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch', 'rock beauty, Holocanthus tricolor', 'anemone fish', 'sturgeon', 'gar, garfish, garpike, billfish, Lepisosteus osseus', 'lionfish', 'puffer, pufferfish, blowfish, globefish', 'abacus', 'abaya', "academic gown, academic robe, judge's robe", 'accordion, piano accordion, squeeze box', 'acoustic guitar', 'aircraft carrier, carrier, flattop, attack aircraft carrier', 'airliner', 'airship, dirigible', 'altar', 'ambulance', 'amphibian, amphibious vehicle', 'analog clock', 'apiary, bee house', 'apron', 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'assault rifle, assault gun', 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'bakery, bakeshop, bakehouse', 'balance beam, beam', 'balloon', 'ballpoint, ballpoint pen, ballpen, Biro', 'Band Aid', 'banjo', 'bannister, banister, balustrade, balusters, handrail', 'barbell', 'barber chair', 'barbershop', 'barn', 'barometer', 'barrel, cask', 'barrow, garden cart, lawn cart, wheelbarrow', 'baseball', 'basketball', 'bassinet', 'bassoon', 'bathing cap, swimming cap', 'bath towel', 'bathtub, bathing tub, bath, tub', 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon', 'beacon, lighthouse, beacon light, pharos', 'beaker', 'bearskin, busby, shako', 'beer bottle', 'beer glass', 'bell cote, bell cot', 'bib', 'bicycle-built-for-two, tandem bicycle, tandem', 'bikini, two-piece', 'binder, ring-binder', 'binoculars, field glasses, opera glasses', 'birdhouse', 'boathouse', 'bobsled, bobsleigh, bob', 'bolo tie, bolo, bola tie, bola', 'bonnet, poke bonnet', 'bookcase', 'bookshop, bookstore, bookstall', 'bottlecap', 'bow', 'bow tie, bow-tie, bowtie', 'brass, memorial tablet, plaque', 'brassiere, bra, bandeau', 'breakwater, groin, groyne, mole, bulwark, seawall, jetty', 'breastplate, aegis, egis', 'broom', 'bucket, pail', 'buckle', 'bulletproof vest', 'bullet train, bullet', 'butcher shop, meat market', 'cab, hack, taxi, taxicab', 'caldron, cauldron', 'candle, taper, wax light', 'cannon', 'canoe', 'can opener, tin opener', 'cardigan', 'car mirror', 'carousel, carrousel, merry-go-round, roundabout, whirligig', "carpenter's kit, tool kit", 'carton', 'car wheel', 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello, violoncello', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'chain', 'chainlink fence', 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour', 'chain saw, chainsaw', 'chest', 'chiffonier, commode', 'chime, bell, gong', 'china cabinet, china closet', 'Christmas stocking', 'church, church building', 'cinema, movie theater, movie theatre, movie house, picture palace', 'cleaver, meat cleaver, chopper', 'cliff dwelling', 'cloak', 'clog, geta, patten, sabot', 'cocktail shaker', 'coffee mug', 'coffeepot', 'coil, spiral, volute, whorl, helix', 'combination lock', 'computer keyboard, keypad', 'confectionery, confectionary, candy store', 'container ship, containership, container vessel', 'convertible', 'corkscrew, bottle screw', 'cornet, horn, trumpet, trump', 'cowboy boot', 'cowboy hat, ten-gallon hat', 'cradle', 'crane', 'crash helmet', 'crate', 'crib, cot', 'Crock Pot', 'croquet ball', 'crutch', 'cuirass', 'dam, dike, dyke', 'desk', 'desktop computer', 'dial telephone, dial phone', 'diaper, nappy, napkin', 'digital clock', 'digital watch', 'dining table, board', 'dishrag, dishcloth', 'dishwasher, dish washer, dishwashing machine', 'disk brake, disc brake', 'dock, dockage, docking facility', 'dogsled, dog sled, dog sleigh', 'dome', 'doormat, welcome mat', 'drilling platform, offshore rig', 'drum, membranophone, tympan', 'drumstick', 'dumbbell', 'Dutch oven', 'electric fan, blower', 'electric guitar', 'electric locomotive', 'entertainment center', 'envelope', 'espresso maker', 'face powder', 'feather boa, boa', 'file, file cabinet, filing cabinet', 'fireboat', 'fire engine, fire truck', 'fire screen, fireguard', 'flagpole, flagstaff', 'flute, transverse flute', 'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster', 'freight car', 'French horn, horn', 'frying pan, frypan, skillet', 'fur coat', 'garbage truck, dustcart', 'gasmask, respirator, gas helmet', 'gas pump, gasoline pump, petrol pump, island dispenser', 'goblet', 'go-kart', 'golf ball', 'golfcart, golf cart', 'gondola', 'gong, tam-tam', 'gown', 'grand piano, grand', 'greenhouse, nursery, glasshouse', 'grille, radiator grille', 'grocery store, grocery, food market, market', 'guillotine', 'hair slide', 'hair spray', 'half track', 'hammer', 'hamper', 'hand blower, blow dryer, blow drier, hair dryer, hair drier', 'hand-held computer, hand-held microcomputer', 'handkerchief, hankie, hanky, hankey', 'hard disc, hard disk, fixed disk', 'harmonica, mouth organ, harp, mouth harp', 'harp', 'harvester, reaper', 'hatchet', 'holster', 'home theater, home theatre', 'honeycomb', 'hook, claw', 'hoopskirt, crinoline', 'horizontal bar, high bar', 'horse cart, horse-cart', 'hourglass', 'iPod', 'iron, smoothing iron', "jack-o'-lantern", 'jean, blue jean, denim', 'jeep, landrover', 'jersey, T-shirt, tee shirt', 'jigsaw puzzle', 'jinrikisha, ricksha, rickshaw', 'joystick', 'kimono', 'knee pad', 'knot', 'lab coat, laboratory coat', 'ladle', 'lampshade, lamp shade', 'laptop, laptop computer', 'lawn mower, mower', 'lens cap, lens cover', 'letter opener, paper knife, paperknife', 'library', 'lifeboat', 'lighter, light, igniter, ignitor', 'limousine, limo', 'liner, ocean liner', 'lipstick, lip rouge', 'Loafer', 'lotion', 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system', "loupe, jeweler's loupe", 'lumbermill, sawmill', 'magnetic compass', 'mailbag, postbag', 'mailbox, letter box', 'maillot', 'maillot, tank suit', 'manhole cover', 'maraca', 'marimba, xylophone', 'mask', 'matchstick', 'maypole', 'maze, labyrinth', 'measuring cup', 'medicine chest, medicine cabinet', 'megalith, megalithic structure', 'microphone, mike', 'microwave, microwave oven', 'military uniform', 'milk can', 'minibus', 'miniskirt, mini', 'minivan', 'missile', 'mitten', 'mixing bowl', 'mobile home, manufactured home', 'Model T', 'modem', 'monastery', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito net', 'motor scooter, scooter', 'mountain bike, all-terrain bike, off-roader', 'mountain tent', 'mouse, computer mouse', 'mousetrap', 'moving van', 'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook, notebook computer', 'obelisk', 'oboe, hautboy, hautbois', 'ocarina, sweet potato', 'odometer, hodometer, mileometer, milometer', 'oil filter', 'organ, pipe organ', 'oscilloscope, scope, cathode-ray oscilloscope, CRO', 'overskirt', 'oxcart', 'oxygen mask', 'packet', 'paddle, boat paddle', 'paddlewheel, paddle wheel', 'padlock', 'paintbrush', "pajama, pyjama, pj's, jammies", 'palace', 'panpipe, pandean pipe, syrinx', 'paper towel', 'parachute, chute', 'parallel bars, bars', 'park bench', 'parking meter', 'passenger car, coach, carriage', 'patio, terrace', 'pay-phone, pay-station', 'pedestal, plinth, footstall', 'pencil box, pencil case', 'pencil sharpener', 'perfume, essence', 'Petri dish', 'photocopier', 'pick, plectrum, plectron', 'pickelhaube', 'picket fence, paling', 'pickup, pickup truck', 'pier', 'piggy bank, penny bank', 'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate, pirate ship', 'pitcher, ewer', "plane, carpenter's plane, woodworking plane", 'planetarium', 'plastic bag', 'plate rack', 'plow, plough', "plunger, plumber's helper", 'Polaroid camera, Polaroid Land camera', 'pole', 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 'poncho', 'pool table, billiard table, snooker table', 'pop bottle, soda bottle', 'pot, flowerpot', "potter's wheel", 'power drill', 'prayer rug, prayer mat', 'printer', 'prison, prison house', 'projectile, missile', 'projector', 'puck, hockey puck', 'punching bag, punch bag, punching ball, punchball', 'purse', 'quill, quill pen', 'quilt, comforter, comfort, puff', 'racer, race car, racing car', 'racket, racquet', 'radiator', 'radio, wireless', 'radio telescope, radio reflector', 'rain barrel', 'recreational vehicle, RV, R.V.', 'reel', 'reflex camera', 'refrigerator, icebox', 'remote control, remote', 'restaurant, eating house, eating place, eatery', 'revolver, six-gun, six-shooter', 'rifle', 'rocking chair, rocker', 'rotisserie', 'rubber eraser, rubber, pencil eraser', 'rugby ball', 'rule, ruler', 'running shoe', 'safe', 'safety pin', 'saltshaker, salt shaker', 'sandal', 'sarong', 'sax, saxophone', 'scabbard', 'scale, weighing machine', 'school bus', 'schooner', 'scoreboard', 'screen, CRT screen', 'screw', 'screwdriver', 'seat belt, seatbelt', 'sewing machine', 'shield, buckler', 'shoe shop, shoe-shop, shoe store', 'shoji', 'shopping basket', 'shopping cart', 'shovel', 'shower cap', 'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule, slipstick', 'sliding door', 'slot, one-armed bandit', 'snorkel', 'snowmobile', 'snowplow, snowplough', 'soap dispenser', 'soccer ball', 'sock', 'solar dish, solar collector, solar furnace', 'sombrero', 'soup bowl', 'space bar', 'space heater', 'space shuttle', 'spatula', 'speedboat', "spider web, spider's web", 'spindle', 'sports car, sport car', 'spotlight, spot', 'stage', 'steam locomotive', 'steel arch bridge', 'steel drum', 'stethoscope', 'stole', 'stone wall', 'stopwatch, stop watch', 'stove', 'strainer', 'streetcar, tram, tramcar, trolley, trolley car', 'stretcher', 'studio couch, day bed', 'stupa, tope', 'submarine, pigboat, sub, U-boat', 'suit, suit of clothes', 'sundial', 'sunglass', 'sunglasses, dark glasses, shades', 'sunscreen, sunblock, sun blocker', 'suspension bridge', 'swab, swob, mop', 'sweatshirt', 'swimming trunks, bathing trunks', 'swing', 'switch, electric switch, electrical switch', 'syringe', 'table lamp', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'tape player', 'teapot', 'teddy, teddy bear', 'television, television system', 'tennis ball', 'thatch, thatched roof', 'theater curtain, theatre curtain', 'thimble', 'thresher, thrasher, threshing machine', 'throne', 'tile roof', 'toaster', 'tobacco shop, tobacconist shop, tobacconist', 'toilet seat', 'torch', 'totem pole', 'tow truck, tow car, wrecker', 'toyshop', 'tractor', 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi', 'tray', 'trench coat', 'tricycle, trike, velocipede', 'trimaran', 'tripod', 'triumphal arch', 'trolleybus, trolley coach, trackless trolley', 'trombone', 'tub, vat', 'turnstile', 'typewriter keyboard', 'umbrella', 'unicycle, monocycle', 'upright, upright piano', 'vacuum, vacuum cleaner', 'vase', 'vault', 'velvet', 'vending machine', 'vestment', 'viaduct', 'violin, fiddle', 'volleyball', 'waffle iron', 'wall clock', 'wallet, billfold, notecase, pocketbook', 'wardrobe, closet, press', 'warplane, military plane', 'washbasin, handbasin, washbowl, lavabo, wash-hand basin', 'washer, automatic washer, washing machine', 'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle', 'wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle', 'wing', 'wok', 'wooden spoon', 'wool, woolen, woollen', 'worm fence, snake fence, snake-rail fence, Virginia fence', 'wreck', 'yawl', 'yurt', 'web site, website, internet site, site', 'comic book', 'crossword puzzle, crossword', 'street sign', 'traffic light, traffic signal, stoplight', 'book jacket, dust cover, dust jacket, dust wrapper', 'menu', 'plate', 'guacamole', 'consomme', 'hot pot, hotpot', 'trifle', 'ice cream, icecream', 'ice lolly, lolly, lollipop, popsicle', 'French loaf', 'bagel, beigel', 'pretzel', 'cheeseburger', 'hotdog, hot dog, red hot', 'mashed potato', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini, courgette', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber, cuke', 'artichoke, globe artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple, ananas', 'banana', 'jackfruit, jak, jack', 'custard apple', 'pomegranate', 'hay', 'carbonara', 'chocolate sauce, chocolate syrup', 'dough', 'meat loaf, meatloaf', 'pizza, pizza pie', 'potpie', 'burrito', 'red wine', 'espresso', 'cup', 'eggnog', 'alp', 'bubble', 'cliff, drop, drop-off', 'coral reef', 'geyser', 'lakeside, lakeshore', 'promontory, headland, head, foreland', 'sandbar, sand bar', 'seashore, coast, seacoast, sea-coast', 'valley, vale', 'volcano', 'ballplayer, baseball player', 'groom, bridegroom', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 'corn', 'acorn', 'hip, rose hip, rosehip', 'buckeye, horse chestnut, conker', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn, carrion fungus', 'earthstar', 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa', 'bolete', 'ear, spike, capitulum', 'toilet tissue, toilet paper, bathroom tissue']
        else:
            raise ValueError
    else:
        raise ValueError
    return dataset1, dataset2

class SubSet(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        labels = torch.LongTensor([ self.dataset.targets[i] for i in indices ] )
        labels_hold = torch.ones(len(dataset)).type(torch.long) *10000 #( some number not present in the #labels just to make sure
        labels_hold[self.indices] = labels 
        self.labels = labels_hold
    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        label = self.labels[self.indices[idx]]
        return (image, label)
    def update_labels(self, new_labels):
        labels_hold = torch.ones(len(self.dataset)).type(torch.long) *10000 #( some number not present in the #labels just to make sure
        labels_hold[self.indices] = torch.LongTensor(new_labels)
        self.labels = labels_hold


    def __len__(self):
        return len(self.indices)

def get_retain_forget_partition(args, dataset, unlearn_class_list, return_ind = False):
    retain_ind = []
    forget_ind = []
    for sample_index in range(len(dataset)) :
        if (torch.is_tensor(dataset.targets[sample_index])):
            sample_class = int(dataset.targets[sample_index].item())
        elif isinstance(dataset.targets[sample_index], int ):
            sample_class = dataset.targets[sample_index]
        if sample_class in unlearn_class_list:
            forget_ind.append(sample_index)
        else:
            retain_ind.append(sample_index)
    retain_dataset = SubSet(dataset, retain_ind)
    forget_dataset = SubSet(dataset, forget_ind)
    if return_ind:
        return retain_dataset, forget_dataset, retain_ind, forget_ind

    return retain_dataset, forget_dataset

def get_model(args,device):
    # Instantiate model
    if "vgg11_bn" in args.arch.lower():
        model = vgg11_bn(num_classes=args.num_classes, dataset=args.dataset).to(device)
    elif "vgg13_bn" in args.arch.lower():
        model = vgg13_bn(num_classes=args.num_classes, dataset=args.dataset).to(device)
    elif "vgg16_bn" in args.arch.lower():
        model = vgg16_bn(num_classes=args.num_classes, dataset=args.dataset).to(device)
    elif "vgg19_bn" in args.arch.lower():
        model = vgg19_bn(num_classes=args.num_classes, dataset=args.dataset).to(device)
    elif "vgg11" in args.arch.lower():
        model = vgg11(num_classes=args.num_classes, dataset=args.dataset).to(device)
    elif "vgg13" in args.arch.lower():
        model = vgg13(num_classes=args.num_classes, dataset=args.dataset).to(device)
    elif "vgg16" in args.arch.lower():
        model = vgg16(num_classes=args.num_classes, dataset=args.dataset).to(device)
    elif "vgg19" in args.arch.lower():
        model = vgg19(num_classes=args.num_classes, dataset=args.dataset).to(device)
    elif "resnet152" in args.arch.lower():
        model = ResNet152(num_classes=args.num_classes, dataset=args.dataset).to(device)
    elif "resnet101" in args.arch.lower():
        model = ResNet101(num_classes=args.num_classes, dataset=args.dataset).to(device)
    elif "resnet50" in args.arch.lower():
        model = ResNet50(num_classes=args.num_classes, dataset=args.dataset).to(device)
    elif "resnet34" in args.arch.lower():
        model = ResNet34(num_classes=args.num_classes, dataset=args.dataset).to(device)
    elif "resnet18" in args.arch.lower():
        model = ResNet18(num_classes=args.num_classes, dataset=args.dataset).to(device)
    elif "vit_b_16" in args.arch.lower():
        model = vit_b_16().to(device)
    elif "vit_b_32" in args.arch.lower():
        model = vit_b_32().to(device)
    elif "vit_l_16" in args.arch.lower():
        model = vit_l_16().to(device)
    elif "vit_l_32" in args.arch.lower():
        model = vit_l_32().to(device)
    elif "vit_h_14" in args.arch.lower():
        model = vit_h_14().to(device)
    else:
        raise ValueError
    return model

    
def metric_function(x, y):
    out= x *(1 - y)
    return out
    
def test(model, device, data_loader,  unlearn_class_list, class_label_names, num_classes =10, \
                    plot_cm=False, job_name = "baseline", verbose=True, set_name = "Val Set"):
    model.eval()
    sample_loss = 0
    correct = 0
    cm = np.zeros((num_classes,num_classes))
    dict_classwise_acc={}
    dict_classwise_loss=defaultdict(float)
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            sample_loss = F.nll_loss(output, target, reduction='none')#.item()  # sum up batch loss
            for i in range(num_classes):
                dict_classwise_loss[i] +=  torch.where(target == i, sample_loss, torch.zeros_like(sample_loss)).sum()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            cm+=confusion_matrix(target.cpu().numpy(),pred.squeeze(-1).cpu().numpy(), labels=[val for val in range(num_classes)])
            correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss = sum(list(dict_classwise_loss.values()))
    total_loss /= len(data_loader.dataset)    
    classwise_acc = cm.diagonal()/cm.sum(axis=1)
    for i in range(0,num_classes):
        dict_classwise_acc[class_label_names[i]] =  100*classwise_acc[i]
    if unlearn_class_list:
        forget_loss = sum([float(dict_classwise_loss[key]) for key in dict_classwise_loss if key in unlearn_class_list])/ sum([float(val) for i, val in enumerate(cm.sum(axis=1)) if i in unlearn_class_list])
        forget_acc = sum([float(val) for i, val in enumerate(cm.diagonal()) if i in unlearn_class_list])/ sum([float(val) for i, val in enumerate(cm.sum(axis=1)) if i in unlearn_class_list])
        unlearn_class_name = [name for i, name in enumerate(class_label_names) if i in unlearn_class_list]
    else:
        forget_acc = 0
        forget_loss = 0
        unlearn_class_name = []
    retain_acc = sum([float(val) for i, val in enumerate(cm.diagonal()) if i not in unlearn_class_list])/ sum([float(val)  for i, val in enumerate(cm.sum(axis=1)) if i not in unlearn_class_list])
    retain_loss = sum([float(dict_classwise_loss[key]) for key in dict_classwise_loss if key not in unlearn_class_list])/ sum([float(val) for i, val in enumerate(cm.sum(axis=1)) if i not in unlearn_class_list])
    metric = metric_function(retain_acc,forget_acc)
    if plot_cm:
        fig,ax = plt.subplots()
        fig.set_size_inches(10,10)
        plt.style.use("seaborn-talk")
        cs = sns.color_palette("muted")
        short_labels = []
        for label in class_label_names:
            short_labels.append(label[0:4])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_labels )
        disp.plot(cmap='Greens', values_format='.0f')
        for labels in disp.text_.ravel():
            labels.set_fontsize(20)
        # plt.title(f"{set_name} Class removed {unlearn_class_name} : {job_name}")
        plt.xticks(rotation=45, ha='right', fontsize=25)
        plt.yticks(rotation=45, fontsize=25)
        plt.xlabel("Predicted Labels", fontsize=30)
        plt.ylabel("True Lables", fontsize=30)
        plt.tight_layout()
        plt.savefig(f"./class_removal/images/cm_{set_name}_{unlearn_class_name}_{job_name}.pdf")
        
        # plt.show()    

    print(f'{set_name}: Average loss: {total_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({(100. * correct / len(data_loader.dataset)):.0f}%)')
    if verbose:
        print('-'*30)
        print(f"{set_name} Confusion Matrix \n{cm}")
        print('-'*30)  
        print(f"{set_name} Class Wise ACC \n{dict_classwise_acc}")
        print('-'*30) 
        print(f"{set_name} Retain class acc(loss): {100*retain_acc}({retain_loss}) - Forget class acc(loss): {100*forget_acc}({forget_loss})\n")
        wandb.log({ #"test_loss":test_loss, 
                    f"{set_name}/acc-forget":100. * forget_acc,
                    f"{set_name}/acc-retained":100. * retain_acc,
                    f"{set_name}/metric":100*(metric) ,
                    f"{set_name}/loss-forget":forget_loss,
                    f"{set_name}/loss-retained":retain_loss,
                    f"{set_name}/class-acc/test_acc":dict_classwise_acc,
                    f"{set_name}/class-loss/test_loss":dict_classwise_loss
                    }
                    )
    return retain_acc, forget_acc, metric




### Simple MIA
def collect_prob(data_loader, model):
    if data_loader is None:
        return torch.zeros([0, 10]), torch.zeros([0])

    prob = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(model.parameters()).device)
                     for tensor in batch]
            data, target = batch

            with torch.no_grad():
                log_prob = model(data) # Returns log_prob. exp( ) 
                prob.append(torch.exp(log_prob).data)
                targets.append(target)

    return torch.cat(prob), torch.cat(targets)


def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0]
    n_target_test = target_test.shape[0]

    X_shadow = torch.cat([shadow_train, shadow_test]).cpu(
    ).numpy().reshape(n_shadow_train + n_shadow_test, -1)
    Y_shadow = np.concatenate(
        [np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma='auto', kernel='rbf')
    clf.fit(X_shadow, Y_shadow)

    accs = []

    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        accs.append(acc_train)

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.append(acc_test)

    return np.mean(accs)


def SVC_MIA(shadow_train, target_train, target_test, shadow_test, model):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model)

    target_train_prob, target_train_labels = collect_prob(target_train, model)
    target_test_prob, target_test_labels = collect_prob(target_test, model)

    shadow_train_conf = torch.gather(
        shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(
        shadow_test_prob, 1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(
        target_train_prob, 1, target_train_labels[:, None])
    target_test_conf = torch.gather(
        target_test_prob, 1, target_test_labels[:, None])

    acc_conf = SVC_fit_predict(
        shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf)
    m = {
         "confidence": acc_conf,
         }
    print(m)
    return m
