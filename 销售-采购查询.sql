SELECT cheku_spec_order.create_time,cheku_spec_order.brand,cheku_spec_order.series,cheku_spec_order.buy_num,cheku_spec_order.username,cheku_spec_order.cheku_price,
		cheku_spec_order.model,cheku_spec_order.transport_method,cheku_spec_order.out_color,cheku_procurement_order.supplyGoods,cheku_procurement_order.city,cheku_procurement_order.prices
FROM 
	chekumaster.cheku_spec_order left join chekumaster.cheku_procurement_order
    on cheku_spec_order.id = cheku_procurement_order.orderIds
where 
	chekumaster.cheku_spec_order.del_flag != 1
    and
    chekumaster.cheku_spec_order.create_time > '2017-01-01';