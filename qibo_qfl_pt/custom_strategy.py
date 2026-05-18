import json
import random
from pathlib import Path
from typing import Optional, Callable
from flwr.app import MetricRecord, ArrayRecord, ConfigRecord, RecordDict, Message
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, FedAdam, FedProx, FedAdagrad, FedYogi, Result
from flwr.common.logger import log
from logging import INFO, WARNING

class StrategyWithMetrics:
    """Mixin che aggiunge salvataggio, stampa metriche e selezione deterministica dei client."""
    
    def __init__(self, save_path: str = "results", suffix: str = "", run_info: dict | None = None,
                 sampling_seed: int = 42, training_mode: str = "federated", noise_info: dict | None = None,
                 nshots=None, model_type: str = "quantum", **kwargs):
        super().__init__(**kwargs)
        self.training_mode = training_mode
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        self.run_info = run_info or {}
        self.run_info["training_mode"] = training_mode
        self.run_info["nshots"] = nshots
        self.run_info["model_type"] = model_type
        if noise_info:
            self.run_info.update(noise_info)
        self.sampling_seed = sampling_seed
        
        # Inizializza variabili per campionamento deterministico
        self.current_round = 0
        self.deterministic_indices = []  # Ordine deterministico degli INDICI
        self.node_id_to_index = {}       # Mapping node_id -> indice fisso
        self.index_to_node_id = {}       # Mapping indice -> node_id
        self.is_initialized = False
        
        strategy_name = self.__class__.__name__.lower()
        if suffix:
            self.metrics_filename = self.save_path / f"{strategy_name}{suffix}.json"
        else:
            self.metrics_filename = self.save_path / f"{strategy_name}.json"
    
    def summary(self) -> None:
        """Log summary con training mode e iperparametri."""
        log(INFO, "\t├──> Model type: %s", self.run_info.get("model_type", "quantum"))
        log(INFO, "\t├──> Training mode: %s", self.training_mode)
        log(INFO, "\t├──> Run info:")
        for key, val in self.run_info.items():
            if val is not None:
                log(INFO, "\t│\t├── %s: %s", key, val)
        super().summary()

    def _initialize_deterministic_order(self, grid: Grid):
        """Inizializza l'ordine deterministico basato su indici fissi."""
        if not self.is_initialized:
            all_node_ids = sorted(list(grid.get_node_ids()))
            num_clients = len(all_node_ids)
            
            # Crea mapping fisso: node_id -> indice (0, 1, 2, ...)
            for idx, node_id in enumerate(all_node_ids):
                self.node_id_to_index[node_id] = idx
                self.index_to_node_id[idx] = node_id
            
            # Crea sequenza deterministicamente shufflata di INDICI
            self.deterministic_indices = list(range(num_clients))
            random.seed(self.sampling_seed)
            random.shuffle(self.deterministic_indices)
            
            self.is_initialized = True

    
    def _get_deterministic_node_ids(self, grid: Grid, fraction: float, min_nodes: int) -> list[int]:
        """Seleziona deterministicamente i client in base al round corrente."""
        # Inizializza l'ordine solo al primo round
        self._initialize_deterministic_order(grid)
        
        # Parametri per la selezione
        num_clients = len(self.deterministic_indices)
        clients_per_round = max(int(num_clients * fraction), min_nodes)
        
        # Campionamento circolare sulla sequenza shufflata di indici
        # Round 1: prende i primi clients_per_round dalla sequenza shufflata
        # Round 2: shifta di 1 nella sequenza shufflata
        start_pos = (self.current_round - 1) % num_clients
        selected_positions = [(start_pos + i) % num_clients for i in range(clients_per_round)]
        
        # Ottieni gli indici dalla sequenza shufflata
        selected_indices = [self.deterministic_indices[pos] for pos in selected_positions]
        
        # Converti indici -> node_ids
        selected_node_ids = [self.index_to_node_id[idx] for idx in selected_indices]
        
        #log(INFO, f"Round {self.current_round}: positions {selected_positions} -> indices {selected_indices} -> node_ids {selected_node_ids}")
        
        return selected_node_ids
    
    def configure_train(
    self,
    server_round: int,
    arrays: ArrayRecord,
    config: ConfigRecord,
    grid: Grid,
):
        """Override per selezione deterministica dei client."""
        if self.fraction_train == 0.0:
            return []
        
        # Selezione deterministica
        node_ids = self._get_deterministic_node_ids(
            grid, 
            self.fraction_train, 
            self.min_train_nodes
        )
        
        # Crea messaggi con partition_id deterministico nel config
        messages = []
        for node_id in node_ids:
            # Ottieni l'indice fisso per questo node_id
            client_index = self.node_id_to_index[node_id]
            
            # Aggiungi partition_id al config
            config_dict = dict(config)
            config_dict["partition_id"] = client_index  # Usa indice fisso come partition_id
            if hasattr(self, "proximal_mu"):
                config_dict["proximal-mu"] = self.proximal_mu
            updated_config = ConfigRecord(config_dict)

            content = RecordDict({
                self.arrayrecord_key: arrays,
                self.configrecord_key: updated_config,
            })

            message = Message(
                content=content,
                dst_node_id=node_id,
                message_type="train",
                group_id=str(server_round),
            )
            messages.append(message)

        return messages

    
    def configure_evaluate(
    self,
    server_round: int,
    arrays: ArrayRecord,
    config: ConfigRecord,
    grid: Grid,
):
        """Override per selezione deterministica dei client (evaluation)."""
        if self.fraction_evaluate == 0.0:
            return []
        
        # Usa gli stessi client del training
        node_ids = self._get_deterministic_node_ids(
            grid, 
            self.fraction_evaluate, 
            self.min_evaluate_nodes
        )
        
        # Crea messaggi con partition_id deterministico nel config
        messages = []
        for node_id in node_ids:
            # Ottieni l'indice fisso per questo node_id
            client_index = self.node_id_to_index[node_id]
            
            # Aggiungi partition_id al config
            config_dict = dict(config)
            config_dict["partition_id"] = client_index
            updated_config = ConfigRecord(config_dict)
            
            content = RecordDict({
                self.arrayrecord_key: arrays,
                self.configrecord_key: updated_config,
            })
            
            message = Message(
                content=content,
                dst_node_id=node_id,
                message_type="evaluate",
                group_id=str(server_round),
            )
            messages.append(message)
        
        return messages


    def start(
    self,
    grid: Grid,
    initial_arrays: ArrayRecord,
    num_rounds: int = 3,
    timeout: float = 3600,
    train_config: Optional[ConfigRecord] = None,
    evaluate_config: Optional[ConfigRecord] = None,
    evaluate_fn: Optional[Callable[[int, ArrayRecord], Optional[MetricRecord]]] = None,
) -> Result:
        """Esegue la strategia con campionamento deterministico."""
        
        log(INFO, f"Starting {self.training_mode} training with {self.__class__.__name__}")
        self.summary()
        
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        
        result = Result()
        arrays = initial_arrays
        
        # Inizializza lo stato della strategia (importante per FedAdam, FedYogi, ecc.)
        # Converti ArrayRecord in dizionario di numpy arrays
        self.current_arrays = {
            k: v for k, v in zip(
                initial_arrays.keys(),
                initial_arrays.to_numpy_ndarrays()
            )
        }
        
        # Inizializza il file JSON
        with open(self.metrics_filename, "w") as f:
            json.dump({"info": self.run_info, "rounds": []}, f, indent=3)
        
        # Round 0 (valutazione server)
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res
                self._save_round_metrics(0, result)
        
        # Ciclo dei Round
        for self.current_round in range(1, num_rounds + 1):
            log(INFO, f"\n[ROUND {self.current_round}/{num_rounds}]")
            
            # Training
            train_messages = self.configure_train(self.current_round, arrays, train_config, grid)
            train_replies = grid.send_and_receive(messages=train_messages, timeout=timeout)
            
            agg_arrays, agg_train_metrics = self.aggregate_train(self.current_round, train_replies)
            if agg_arrays is not None:
                arrays = agg_arrays
                result.arrays = agg_arrays
                # Aggiorna current_arrays per strategie adaptive
                self.current_arrays = {
                    k: v for k, v in zip(
                        agg_arrays.keys(),
                        agg_arrays.to_numpy_ndarrays()
                    )
                }
            if agg_train_metrics is not None:
                result.train_metrics_clientapp[self.current_round] = agg_train_metrics
                log(INFO, f" └──> Train metrics: {dict(agg_train_metrics)}")
            
            # Evaluation
            if self.fraction_evaluate > 0:
                evaluate_messages = self.configure_evaluate(self.current_round, arrays, evaluate_config, grid)
                evaluate_replies = grid.send_and_receive(messages=evaluate_messages, timeout=timeout)
                
                agg_evaluate_metrics = self.aggregate_evaluate(self.current_round, evaluate_replies)
                if agg_evaluate_metrics is not None:
                    result.evaluate_metrics_clientapp[self.current_round] = agg_evaluate_metrics
                    log(INFO, f" └──> Eval metrics: {dict(agg_evaluate_metrics)}")
            
            # Valutazione server
            if evaluate_fn:
                res = evaluate_fn(self.current_round, arrays)
                if res is not None:
                    result.evaluate_metrics_serverapp[self.current_round] = res
            
            # Salvataggio metriche
            self._save_round_metrics(self.current_round, result)
        
        log(INFO, f"\nTraining completed! Metrics saved in {self.save_path}")
        return result


    
    def _save_round_metrics(self, current_round: int, result: Result):
        round_data = {
            "round": current_round,
            "train_metrics": dict(result.train_metrics_clientapp.get(current_round, {})),
            "eval_metrics_client": dict(result.evaluate_metrics_clientapp.get(current_round, {})),
            "eval_metrics_server": dict(result.evaluate_metrics_serverapp.get(current_round, {})),
        }
        
        self.metrics_history.append(round_data)
        
        with open(self.metrics_filename, "r+") as f:
            data = json.load(f)
            data["rounds"] = self.metrics_history
            f.seek(0)
            json.dump(data, f, indent=3)
            f.truncate()


class fedavg(StrategyWithMetrics, FedAvg):
    """FedAvg con salvataggio e stampa delle metriche."""
    def __init__(self, seed, num_rounds, num_clients, local_epochs, iid, alpha,
                 eta_l=None, sampling_seed=None, **kwargs):
        run_info = {
            "strategy": "FedAvg",
            "seed": seed,
            "sampling_seed": sampling_seed if sampling_seed is not None else seed,
            "fraction_train": kwargs.get("fraction_train", 1.0),
            "fraction_evaluate": kwargs.get("fraction_evaluate", 1.0),
            "local_epochs": local_epochs,
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "eta_l": eta_l,
            "iid": iid,
            "alpha": alpha if not iid else None,
        }
        suffix = f"_etal{eta_l}_seed{seed}"
        super().__init__(
            suffix=suffix,
            run_info=run_info,
            sampling_seed=sampling_seed if sampling_seed is not None else seed,
            **kwargs,
        )


class fedadam(StrategyWithMetrics, FedAdam):
    """FedAdam con salvataggio e stampa delle metriche."""
    def __init__(self, seed, eta, eta_l, num_rounds, num_clients, local_epochs, iid, alpha,
                 sampling_seed=None, **kwargs):
        run_info = {
            "strategy": "FedAdam",
            "seed": seed,
            "sampling_seed": sampling_seed if sampling_seed is not None else seed,
            "fraction_train": kwargs.get("fraction_train", 1.0),
            "fraction_evaluate": kwargs.get("fraction_evaluate", 1.0),
            "local_epochs": local_epochs,
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "eta": eta,
            "eta_l": eta_l,
            "iid": iid,
            "alpha": alpha if not iid else None,
        }
        suffix = f"_eta{eta}_etal{eta_l}_seed{seed}"
        super().__init__(
            eta=eta,
            eta_l=eta_l,
            suffix=suffix,
            run_info=run_info,
            sampling_seed=sampling_seed if sampling_seed is not None else seed,
            **kwargs,
        )


class fedadagrad(StrategyWithMetrics, FedAdagrad):
    def __init__(self, seed, eta, eta_l, num_rounds, num_clients, local_epochs, iid, alpha,
                 sampling_seed=None, **kwargs):
        run_info = {
            "strategy": "FedAdagrad",
            "seed": seed,
            "sampling_seed": sampling_seed if sampling_seed is not None else seed,
            "local_epochs": local_epochs,
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "eta": eta,
            "eta_l": eta_l,
            "iid": iid,
            "alpha": alpha if not iid else None,
        }
        suffix = f"_eta{eta}_etal{eta_l}_seed{seed}"
        super().__init__(
            eta=eta,
            eta_l=eta_l,
            suffix=suffix,
            run_info=run_info,
            sampling_seed=sampling_seed if sampling_seed is not None else seed,
            **kwargs,
        )


class fedyogi(StrategyWithMetrics, FedYogi):
    def __init__(self, seed, eta, eta_l, num_rounds, num_clients, local_epochs, iid, alpha,
                 sampling_seed=None, **kwargs):
        run_info = {
            "strategy": "FedYogi",
            "seed": seed,
            "sampling_seed": sampling_seed if sampling_seed is not None else seed,
            "local_epochs": local_epochs,
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "eta": eta,
            "eta_l": eta_l,
            "iid": iid,
            "alpha": alpha if not iid else None,
        }
        suffix = f"_eta{eta}_etal{eta_l}_seed{seed}"
        super().__init__(
            eta=eta,
            eta_l=eta_l,
            suffix=suffix,
            run_info=run_info,
            sampling_seed=sampling_seed if sampling_seed is not None else seed,
            **kwargs,
        )


class fedprox(StrategyWithMetrics, FedProx):
    """FedProx con salvataggio e stampa delle metriche."""
    def __init__(self, seed, mu, num_rounds, num_clients, iid, alpha, local_epochs,
                 eta_l=None, sampling_seed=None, **kwargs):
        suffix = f"_mu{mu}_etal{eta_l}_seed{seed}"
        run_info = {
            "strategy": "FedProx",
            "seed": seed,
            "sampling_seed": sampling_seed if sampling_seed is not None else seed,
            "local_epochs": local_epochs,
            "mu": mu,
            "eta_l": eta_l,
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "iid": iid,
            "alpha": alpha if not iid else None,
        }
        super().__init__(
            proximal_mu=mu,
            suffix=suffix,
            run_info=run_info,
            sampling_seed=sampling_seed if sampling_seed is not None else seed,
            **kwargs,
        )