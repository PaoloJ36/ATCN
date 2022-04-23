import data
import config
import process_handler as handler


def main():
    cfg = config.Config()

    # Load or create the data
    apple = data.Data(company="Apple", method=2, config=cfg, mode='Load')
    heineken = data.Data(company="Heineken", method=2, config=cfg, mode='Load')
    postnl = data.Data(company="PostNL", method=2, config=cfg, mode='Load')

    if cfg.experiment1:
        """Run the experiment for each company"""
        models, histories_val, results = handler.experiment1(config=cfg, train=apple.window.train, val=apple.window.val,
                                                             test=apple.window.test,
                                                             setting='Save', stock='Apple')
        models2, histories_val2, results2 = handler.experiment1(config=cfg, train=postnl.window.train,
                                                                val=postnl.window.val, test=postnl.window.test,
                                                                setting='Save', stock='PostNL')
        models3, histories_val3, results3 = handler.experiment1(config=cfg, train=heineken.window.train,
                                                                val=heineken.window.val, test=heineken.window.test,
                                                                setting='Save', stock='Heineken')

    if cfg.experiment2:
        """Create the configs and data windows for 4 and 5 layers, needed because of the different receptive field"""
        cfg4 = config.Config()
        cfg4.layers = 4
        cfg4.kernel_size = 9
        heineken4 = data.Data(company="Heineken", method=2, config=cfg4, mode='Standalone')

        cfg5 = config.Config()
        cfg5.layers = 5
        cfg5.kernel_size = 5
        heineken5 = data.Data(company="Heineken", method=2, config=cfg5, mode='Standalone')

        """Run the experiment for each model"""
        models4, histories_val4, results4 = handler.experiment2(config=cfg, train=heineken.window.train,
                                                                val=heineken.window.val, test=heineken.window.test,
                                                                stock='Heineken', model_name='TCN', model_index=0,
                                                                train4=heineken4.window.train,
                                                                val4=heineken4.window.val, test4=heineken4.window.test,
                                                                train5=heineken5.window.train,
                                                                val5=heineken5.window.val, test5=heineken5.window.test)
        models5, histories_val5, results5 = handler.experiment2(config=cfg, train=heineken.window.train,
                                                                val=heineken.window.val, test=heineken.window.test,
                                                                stock='Heineken', model_name='HATCN', model_index=1,
                                                                train4=heineken4.window.train,
                                                                val4=heineken4.window.val, test4=heineken4.window.test,
                                                                train5=heineken5.window.train,
                                                                val5=heineken5.window.val, test5=heineken5.window.test)
        models6, histories_val6, results6 = handler.experiment2(config=cfg, train=heineken.window.train,
                                                                val=heineken.window.val, test=heineken.window.test,
                                                                stock='Heineken', model_name='TCAN', model_index=2,
                                                                train4=heineken4.window.train,
                                                                val4=heineken4.window.val, test4=heineken4.window.test,
                                                                train5=heineken5.window.train,
                                                                val5=heineken5.window.val, test5=heineken5.window.test)
        models7, histories_val7, results7 = handler.experiment2(config=cfg, train=heineken.window.train,
                                                                val=heineken.window.val, test=heineken.window.test,
                                                                stock='Heineken', model_name='ATCN', model_index=3,
                                                                train4=heineken4.window.train,
                                                                val4=heineken4.window.val, test4=heineken4.window.test,
                                                                train5=heineken5.window.train,
                                                                val5=heineken5.window.val, test5=heineken5.window.test)


main()

