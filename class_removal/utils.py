
from __future__ import print_function
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
    retain_dataset = torch.utils.data.Subset(dataset, retain_ind)
    forget_dataset = torch.utils.data.Subset(dataset, forget_ind)
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

def get_representation_matrix_class(net, device, data_loader, class_labels=[], num_classes = 10,
                               samples_per_class=150, max_batch_size=150, max_samples=50000, set_name = "Retain Set"): 
    if class_labels: 
        # sort data per class and collect samples from required class.
        samples_list = []
        samples_per_class_count = defaultdict(int)
        total_classes = len(class_labels)
        pbar = tqdm(data_loader)
        for data, target in pbar:
            pbar.set_description(f"Collecting Subset of Samples {set_name}:{sum(list(samples_per_class_count.values()))}/{samples_per_class*total_classes}")
            y_sorted, indices = target.sort()
            indices = torch.LongTensor(indices)
            x_sorted = data[indices]
            sample_counts = Counter(y_sorted.cpu().numpy())
            for class_number in range(num_classes):
                if class_number in class_labels:
                    if samples_per_class_count[class_number] == samples_per_class:
                        continue
                    elif samples_per_class_count[class_number] + sample_counts[class_number] > samples_per_class:
                        samples_list.append(x_sorted[0:samples_per_class - samples_per_class_count[class_number] ])
                        samples_per_class_count[class_number]  = samples_per_class
                    else:
                        samples_list.append(x_sorted[0:sample_counts[class_number]])  
                        samples_per_class_count[class_number] += sample_counts[class_number] 
                if sum(list(samples_per_class_count.values())) == samples_per_class*total_classes:
                    break
                x_sorted = x_sorted[sample_counts[class_number]:]
            if sum(list(samples_per_class_count.values())) == samples_per_class*total_classes:
                pbar.set_description(f"Collecting Subset of Samples {set_name}:{sum(list(samples_per_class_count.values()))}/{samples_per_class*total_classes}")
                break
        sample_tensor = torch.cat(samples_list, 0).to(device)
        # Gets prepresentations as dict of dicts # form dataloader without transform
        activations = None 
        net.eval()
        for batch in tqdm(torch.split(sample_tensor, max_batch_size, dim=0), desc=f"Extracting representation for {set_name}"):
            batch_activations = net.get_activations(batch)
            ### Instantinously compress the batch
            for loc in batch_activations.keys():
                for key in batch_activations[loc].keys():
                    if batch_activations[loc][key].shape[0]> (int(max_samples/(sample_tensor.shape[0]/max_batch_size)) +1):
                        ### Shuffle and return a subset of patches
                        r=np.arange(batch_activations[loc][key].shape[0])
                        np.random.shuffle(r)
                        b = r[:(int(max_samples/(sample_tensor.shape[0]/max_batch_size)) +1)]
                        batch_activations[loc][key] = batch_activations[loc][key][b].copy()
            ### Concatinate the samples
            if activations:
                for loc in batch_activations.keys():
                    for key in batch_activations[loc].keys():
                        activations[loc][key] = np.concatenate([activations[loc][key],batch_activations[loc][key]], 0)
            else:
                activations = batch_activations
        ### Final check for reducing the sample size
        sampled_activations={"pre":OrderedDict(),"post":OrderedDict(),}
        for loc in batch_activations.keys():
            for key in batch_activations[loc].keys():
                if activations[loc][key].shape[0]> max_samples:
                    ### Shuffle and return a subset of patches
                    r=np.arange(activations[loc][key].shape[0])
                    np.random.shuffle(r)
                    b = r[:max_samples]
                    sampled_activations[loc][key] = activations[loc][key][b].copy()
                else:
                    sampled_activations[loc][key] = activations[loc][key].copy()
        # Transpose activations
        loc_keys = list(sampled_activations.keys())
        mat_dict={loc:OrderedDict() for loc in loc_keys}
        for loc in loc_keys:
            for act in list(sampled_activations[loc].keys()):
                activation = sampled_activations[loc][act].transpose()
                mat_dict[loc][act]= activation
        #Prints the representation shapes.
        for loc in loc_keys:
            print('-'*30)
            print(f'Representation Matrix {loc} Layer for {set_name}')
            print('-'*30)    
            for act in list(sampled_activations[loc].keys()):
                print (f' Layer {act} : [{mat_dict[loc][act].shape}]')
            print('-'*30)
        return mat_dict
    else:
        return {loc:OrderedDict() for loc in ["pre", "post"]}
   
def get_SVD (mat_dict,   set_name = "SVD"):
    feature_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    s_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    for loc in mat_dict.keys():
        for act in tqdm(mat_dict[loc].keys(), desc=f"{loc}layer - SVD for {set_name}"):
            activation = torch.Tensor(mat_dict[loc][act]).to("cuda")
            U,S,Vh = torch.linalg.svd(activation, full_matrices=False)
            U = U.cpu().numpy()
            S = S.cpu().numpy()            
            feature_dict[loc][act] = U
            s_dict[loc][act] = S
    return feature_dict,  s_dict

def select_basis(feature_dict, full_s_dict, threshold):
    if threshold is None:
        return feature_dict
    out_feature_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    for loc in feature_dict.keys():
        for act in feature_dict[loc].keys():
            U = feature_dict[loc][act]
            S = full_s_dict[loc][act]
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold) +1  
            out_feature_dict[loc][act] = U[:,:r]
    print('-'*40)
    print(f'Gradient Constraints Summary')
    print('-'*40)
    for loc in range(feature_dict.keys()):
        for act in range(feature_dict[loc].keys()):
            print (f'{loc} layer {act} : {out_feature_dict[act].shape[1]}/{out_feature_dict[act].shape[0]}')
    print('-'*40)
    return out_feature_dict

def get_scaled_feature_mat(feature_dict, full_s_dict, mode, alpha, device):
    feature_mat_dict = {"pre":OrderedDict(), "post":OrderedDict()}
    # Projection Matrix Precomputation
    for loc in feature_dict.keys():
        for act in feature_dict[loc].keys():
            U = torch.Tensor( feature_dict[loc][act] ).to(device)
            S = full_s_dict[loc][act]
            r = U.shape[1]
            if mode == "baseline":
                importance = torch.ones(r).to(device) 
            elif mode == "gpm":
                importance = torch.ones(r).to(device) 
            elif mode == "sgp":
                importance = torch.Tensor(( alpha*S/( (alpha-1)*S+max(S)) )[:r]).to(device) 
            elif mode == "sap":
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                importance =  torch.Tensor(( alpha*sval_ratio/((alpha-1)*sval_ratio+1) ) [:r]).to(device) 
            else:
                raise ValueError
            U.requires_grad = False
            feature_mat_dict[loc][act] = torch.mm( U, torch.diag(importance**0.5) )
    return feature_mat_dict

def get_projections(feature_mat_retain_dict, feature_mat_unlearn_dict,projection_type, device):
    feature_mat = {"pre":OrderedDict(), "post":OrderedDict()}
    for loc in feature_mat_retain_dict.keys():
        for act in feature_mat_retain_dict[loc].keys():
            Ur = feature_mat_retain_dict[loc][act]
            Uf = feature_mat_unlearn_dict[loc][act]  
            Mr = torch.mm(Ur, Ur.transpose(0,1))     
            Mf = torch.mm(Uf, Uf.transpose(0,1))
            I = torch.eye(Mf.shape[0]).to(device) 
            Mri = torch.mm(Mr, Mf) # Intersection in terms of retain space basis
            Mfi = torch.mm(Mf, Mr)  # Intersection in terms of forget space basis
            # Select type of projection. 
            if projection_type == "baseline":
                feature_mat[loc][act]= I 
            elif projection_type == "Mr":
                feature_mat[loc][act]= Mr                       
            elif projection_type == "I-(Mf-Mi)":
                feature_mat[loc][act]= I - (Mf - Mfi)
            # elif projection_type == "I-Mf":
            #     feature_mat[loc][act]= I - Mf   
            # elif projection_type == "Mr-Mi":
            #     feature_mat[loc][act]= Mr - Mri        
            else:
                raise ValueError
    return feature_mat

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
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_label_names )
        disp.plot(cmap='Greens', values_format='.0f')
        for labels in disp.text_.ravel():
            labels.set_fontsize(15)
        # plt.title(f"{set_name} Class removed {unlearn_class_name} : {job_name}")
        plt.xticks(rotation=45, ha='right', fontsize=20)
        plt.yticks(rotation=45, fontsize=20)
        plt.xlabel("Predicted Labels", fontsize=20)
        plt.ylabel("True Lables", fontsize=20)
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
    
def activation_projection_based_unlearning(args, model, train_loader, val_loader, test_loader, device):
    # Initial Evaluation
    if not args.save_model:
        if args.val_set_mode:
            base_ra, base_fa, base_metric = test(model, device, val_loader, args.unlearn_class, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Val Set", verbose=False)
            _ = test(model, device, test_loader, args.unlearn_class, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Test Set", verbose=False)
        else:
            base_ra, base_fa, base_metric = test(model, device, train_loader, args.unlearn_class, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Train Set", verbose=False)
            _ = test(model, device, test_loader, args.unlearn_class, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Test Set", verbose=False)
    else:
        base_metric = 0
    best_model = copy.deepcopy(model)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # Gets the basis vectors for retain set    
    retain_classes = [c for c in range(args.num_classes) if (c not in args.unlearn_class and c not in args.ignore_class) ]
    mat_retain_dict = get_representation_matrix_class(model, device, train_loader, class_labels=retain_classes
                                                                  , num_classes = args.num_classes, samples_per_class=args.retain_samples, max_batch_size=args.max_batch_size, max_samples=args.max_samples, set_name = "Retain Set")  
    # Gets the basis vectors for Forget set
    full_feature_retain_dict, full_s_retain_dict = get_SVD( mat_retain_dict,f"SVD Retain Set") 
    mat_unlearn_dict= get_representation_matrix_class(model, device, train_loader, class_labels=args.unlearn_class # don't use ignore_class here!
                                                                  , num_classes = args.num_classes, samples_per_class=args.forget_samples, max_batch_size=args.max_batch_size, max_samples=args.max_samples, set_name = "Forget Set")  
    full_feature_unlearn_dict, full_s_unlearn_dict= get_SVD( mat_unlearn_dict,f"SVD Forget Set") 
    #update unlearn classes
    args.unlearn_class  = [c for c in range(args.num_classes) if (c in args.unlearn_class or c in args.ignore_class) ]
    num_layer = len(full_feature_retain_dict["pre"])
    for mode in args.mode:
        # Iterate over all the retain modes
        for mode_forget in args.mode_forget:
            # Iterate over all the forget modes
            for projection_type in args.projection_type:
                # Iterate over all the projection types
                for start in args.start_layer:
                    for end in args.end_layer:   
                        # Loop to update feature mat to ignore layers between start and num_layers-end
                        # Set retain eps_threshold and scale_coff_list and get the basis
                        if mode == "baseline":
                            scale_coff_list = [0]
                            eps_threshold = None
                        elif mode == "gpm":
                            scale_coff_list = [0]
                            eps_threshold = args.gpm_eps
                        else:
                            scale_coff_list = args.scale_coff   
                            eps_threshold = None
                        for projection_location in args.projection_location:
                            # Iterate over all the projection location
                            best_metric = base_metric  
                            for alpha in scale_coff_list:                            
                                # Set forget eps_threshold and scale_coff_list and get the basis 
                                if mode_forget is None:
                                    mode_forget = mode
                                    scale_coff_list_forget = [alpha]
                                    eps_threshold_forget = eps_threshold
                                elif mode_forget == "baseline":
                                    scale_coff_list_forget = [0]
                                    eps_threshold_forget = None
                                elif mode_forget == "gpm":
                                    scale_coff_list_forget = [0]
                                    eps_threshold_forget = args.gpm_eps
                                else:
                                    if projection_type =="I-(Mf-Mi)":
                                        scale_coff_list_forget =[val for val in args.scale_coff_forget if alpha/1000<val]
                                    elif projection_type == "Mr" or projection_type == "baseline":
                                        scale_coff_list_forget =[0]
                                    else:
                                        print(projection_type)
                                        raise ValueError
                                    eps_threshold_forget = None
                                # Obtain the feature_matrix for retain space Mr
                                feature_retain_dict  = select_basis(full_feature_retain_dict, full_s_retain_dict, eps_threshold) 
                                feature_mat_retain_dict = get_scaled_feature_mat(feature_retain_dict, full_s_retain_dict, mode, alpha, device)  
                                terminate_alpha = False                             
                                for alpha_forget in scale_coff_list_forget: 
                                    if terminate_alpha: 
                                        # This parameter set at the end of the loop
                                        break
                                    # Set wandb parameters 
                                    if mode==mode_forget=="baseline" and projection_type=="baseline":
                                        job_name = "baseline" 
                                    elif mode==mode_forget=="baseline" and projection_type!="baseline":
                                        continue
                                    elif projection_type=="baseline":
                                        continue
                                    elif (mode=="baseline" and mode_forget!="baseline") or (mode!="baseline" and mode_forget=="baseline"):
                                        continue
                                    else:
                                        job_name = f"{mode}({alpha})"  if not(mode=="baseline" ) else "baseline"
                                        job_name = f"{job_name}-{mode_forget}({alpha_forget})"  if not(mode_forget=="baseline") else f"{job_name}-baseline"
                                        job_name = f"{job_name}:{projection_type}" if not(projection_type =="baseline" or (mode_forget=="baseline" and mode=="baseline")) else "baseline"
                                    job_name = f"{job_name}:{start}-{num_layer-end}"
                                    # Obtain the feature_matrix for forget space Mf
                                    feature_unlearn_dict  = select_basis(full_feature_unlearn_dict, full_s_unlearn_dict, eps_threshold_forget)  
                                    feature_mat_unlearn_dict = get_scaled_feature_mat(feature_unlearn_dict, full_s_unlearn_dict, mode_forget, alpha_forget, device)                    
                                    # Get the projection matrix using Mf and Mr
                                    projection_mat = get_projections(feature_mat_retain_dict, feature_mat_unlearn_dict,projection_type, device)
                                    # Modify the projection matrix to respect the layers and the projection location in consideration (Puts identity when layer/projection location not in consideration)
                                    modified_projection_mat = {"pre":OrderedDict(), "post":OrderedDict()}
                                    for loc in projection_mat.keys():
                                        for i,act in enumerate(projection_mat[loc].keys()):
                                            if i<start:
                                                modified_projection_mat[loc][act] = torch.eye(projection_mat[loc][act].shape[0]).to(device)
                                            elif i>num_layer-end:
                                                modified_projection_mat[loc][act] = torch.eye(projection_mat[loc][act].shape[0]).to(device)
                                            elif projection_location != "all" and (loc!=projection_location): 
                                                modified_projection_mat[loc][act] = torch.eye(projection_mat[loc][act].shape[0]).to(device)
                                            else:
                                                modified_projection_mat[loc][act] = projection_mat[loc][act]
                                    # Copy of the orignial trained model and project its weight.
                                    inference_model = copy.deepcopy(model)
                                    inference_model.project_weights(modified_projection_mat)
                                    if args.save_model:
                                        ra, fa, _ = test(inference_model, device, test_loader, args.unlearn_class, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Test Set", verbose=False)
                                        print(ra, fa)
                                        torch.save(inference_model.state_dict(), f"./class_removal/pretrained_models/main/{args.dataset}_{args.arch}_forget{args.unlearn_class}.pt")
                                        return 
                                    # Instantiate wandb. 
                                    if args.multiclass:
                                        run = wandb.init(
                                            # Set the project where this run will be logged
                                            project=f"Class-{args.dataset}-{args.project_name}",
                                            group= f"{projection_location}layer-{args.group_name}-{args.arch}", 
                                            name=job_name,
                                            # Track hyperparameters and run metadata
                                            config= vars(args))    
                                    else:
                                        run = wandb.init(
                                            # Set the project where this run will be logged
                                            project=f"Class-{args.dataset}-{args.project_name}",
                                            group= f"{projection_location}layer-{args.group_name}-{args.arch}-{args.unlearn_class}", 
                                            name=job_name,
                                            # Track hyperparameters and run metadata
                                            config= vars(args))
                                    print(args.unlearn_class )
                                    # Evaluates the projection. Prints Confusion matrix and returns retain acc and forget acc.
                                    if args.val_set_mode:
                                        ra,fa, metric = test(inference_model, device, val_loader, args.unlearn_class, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Val Set")                                     
                                        _ = test(inference_model, device, test_loader, args.unlearn_class, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Test Set", plot_cm=args.plot_cm, job_name=f"{args.arch}_{job_name}")
                                    else:
                                        ra,fa, metric = test(inference_model, device, train_loader, args.unlearn_class, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Train Set")
                                        _ = test(inference_model, device, test_loader, args.unlearn_class, class_label_names=args.class_label_names, num_classes = args.num_classes, set_name="Test Set", plot_cm=args.plot_cm, job_name=f"{args.arch}_{job_name}")
                                    wandb.finish()                                        
                                    # Search space reduction (Needs baseline acc => the first mode and mode-forget must be set to baseline.)
                                    if metric > best_metric:
                                        best_model = copy.deepcopy(inference_model)
                                        best_metric = metric

                                    if ra < best_metric or (fa < 1/args.num_classes):
                                        # Terminate alpha_forget search as increase in alpha_forget decrease retain acc. (Less than 0.9*base_retain not acceptable)
                                        # Terminate alpha_forget search as minimum accuracy attained. Further increase will reduce retain acc. 
                                        terminate_alpha  = True
                                    else:
                                        continue
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    if args.multiclass:
        run = wandb.init(
                    # Set the project where this run will be logged
                    project=f"Class-{args.dataset}-{args.project_name}",
                    group= f"{projection_location}layer-{args.group_name}-{args.arch}", 
                    name="sim_time",
                    # Track hyperparameters and run metadata
                    config= vars(args)
        )  
    else:
        run = wandb.init(
                    # Set the project where this run will be logged
                    project=f"Class-{args.dataset}-{args.project_name}",
                    group= f"{projection_location}layer-{args.group_name}-{args.arch}-{args.unlearn_class}", 
                    name="sim_time",
                    # Track hyperparameters and run metadata
                    config= vars(args)
        )       
    wandb.log({"run_time":elapsed_time_ms})
    wandb.finish()   
    return best_model
    
    